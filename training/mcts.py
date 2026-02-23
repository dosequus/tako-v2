"""Monte Carlo Tree Search (MCTS) implementation for HRM."""

import numpy as np
import torch
import logging
from typing import Dict, Optional, List
import copy

logger = logging.getLogger(__name__)


class MCTSNode:
    """Node in the MCTS search tree.

    Attributes:
        visit_count: Number of times this node has been visited
        total_value: Sum of all values backed up through this node
        prior: Prior probability from policy network
        children: Dict mapping action -> child MCTSNode
        parent: Parent node (None for root)
    """

    def __init__(self, prior: float = 1.0, parent: Optional['MCTSNode'] = None):
        """Initialize a new MCTS node.

        Args:
            prior: Prior probability from policy network (default: 1.0 for root)
            parent: Parent node (None for root)
        """
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.children: Dict[int, MCTSNode] = {}
        self.parent = parent

    def value(self) -> float:
        """Return average value (Q-value) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children expanded)."""
        return len(self.children) == 0

    def expand(self, legal_moves: List[int], policy_priors: np.ndarray):
        """Expand this node by adding children for all legal moves.

        Args:
            legal_moves: List of legal move indices
            policy_priors: Policy probability distribution over all actions [action_size]
        """
        for action in legal_moves:
            if action not in self.children:
                # Use policy network's prior for this action
                prior = policy_priors[action]
                self.children[action] = MCTSNode(prior=prior, parent=self)


class MCTS:
    """Monte Carlo Tree Search with PUCT exploration.

    Implements AlphaZero-style MCTS with:
    - PUCT formula for action selection
    - Dirichlet noise at root for exploration
    - Temperature-based action sampling
    - Integration with HRM model evaluation
    """

    def __init__(
        self,
        model,
        game_class,
        config: Dict,
        device: str = 'cpu'
    ):
        """Initialize MCTS.

        Args:
            model: HRM model for position evaluation
            game_class: Game class (e.g., OthelloGame)
            config: MCTS configuration dict with keys:
                - simulations: Number of MCTS simulations
                - puct_c: PUCT exploration constant
                - dirichlet_alpha: Dirichlet noise alpha parameter
                - dirichlet_epsilon: Dirichlet noise mixing weight
                - temperature: Temperature for move selection
                - temperature_threshold: Move number after which temp → 0
                - max_segments_inference: Max segments during inference (default: 1)
            device: Device for model inference ('cpu' or 'cuda')
        """
        self.model = model
        self.game_class = game_class
        self.simulations = config['simulations']
        self.puct_c = config['puct_c']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.dirichlet_epsilon = config['dirichlet_epsilon']
        self.temperature = config['temperature']
        self.temperature_threshold = config['temperature_threshold']
        self.max_segments_inference = config.get('max_segments_inference', 1)  # Default to 1
        self.device = device

        # Move model to device
        self.model.to(device)
        self.model.eval()

    def search(self, game, move_num: int = 0) -> np.ndarray:
        """Run MCTS from current game state and return visit count distribution.

        Args:
            game: Game instance (must implement BaseGame interface)
            move_num: Current move number (for temperature schedule)

        Returns:
            Policy target as visit count distribution [action_size]
        """
        # Create root node
        root = MCTSNode()

        # Run simulations
        for _ in range(self.simulations):
            # Copy game state for this simulation
            sim_game = copy.deepcopy(game)
            self._simulate(sim_game, root)

        # Extract policy from visit counts
        policy = self._get_policy_target(root, move_num, game.action_size())

        # Log search completion (debug level to avoid spam)
        logger.debug(
            f"MCTS search complete: move={move_num}, sims={self.simulations}, "
            f"visits={root.visit_count}, legal_moves={len([a for a in range(game.action_size()) if policy[a] > 0])}"
        )

        return policy

    def _simulate(self, game, node: MCTSNode):
        """Run one MCTS simulation: select → expand → backup.

        Args:
            game: Game instance (will be modified during simulation)
            node: Current node in the tree
        """
        # Selection: walk tree using PUCT until we reach a leaf
        path = []  # List of (node, action) pairs

        while not node.is_leaf() and not game.is_terminal():
            action = self._select_action_puct(node, game.legal_moves())
            path.append((node, action))

            # Make move and descend
            game.make_move(action)
            node = node.children[action]

        # Expansion and evaluation
        if game.is_terminal():
            # Terminal state: use actual outcome
            value = game.outcome()
        else:
            # Leaf node: evaluate with network and expand
            legal_moves = game.legal_moves()

            # Get network evaluation
            policy_logits, value_logits = self._evaluate(game)

            # Convert value logits to expected value
            # value_logits is [3] for W/D/L probabilities (log probs)
            value_probs = torch.softmax(value_logits, dim=0)
            # W=+1, D=0, L=-1
            value = float(value_probs[0] * 1.0 + value_probs[1] * 0.0 + value_probs[2] * (-1.0))

            # Expand node with legal moves
            policy_probs = torch.softmax(policy_logits, dim=0).cpu().numpy()

            # Add Dirichlet noise at root for exploration
            if node == path[0][0] if path else True:  # Root node
                policy_probs = self._add_dirichlet_noise(policy_probs, legal_moves)

            node.expand(legal_moves, policy_probs)

        # Backup: propagate value up the tree
        # Value flips sign at each level (zero-sum game)
        for parent_node, action in reversed(path):
            child = parent_node.children[action]
            child.visit_count += 1
            child.total_value += value
            value = -value  # Flip for parent's perspective

    def _select_action_puct(self, node: MCTSNode, legal_moves: List[int]) -> int:
        """Select action using PUCT formula.

        PUCT formula: Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Args:
            node: Current node
            legal_moves: List of legal move indices

        Returns:
            Selected action index
        """
        best_score = -float('inf')
        best_action = legal_moves[0]

        # Parent visit count for exploration term
        sqrt_parent_visits = np.sqrt(node.visit_count)

        for action in legal_moves:
            if action not in node.children:
                # Should not happen after expansion, but handle gracefully
                continue

            child = node.children[action]

            # Q-value (average value)
            q_value = child.value()

            # Exploration bonus: c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            exploration = (
                self.puct_c * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            )

            # PUCT score
            score = q_value + exploration

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _add_dirichlet_noise(
        self,
        policy_probs: np.ndarray,
        legal_moves: List[int]
    ) -> np.ndarray:
        """Add Dirichlet noise to policy at root for exploration.

        Args:
            policy_probs: Policy probability distribution [action_size]
            legal_moves: List of legal move indices

        Returns:
            Noised policy [action_size]
        """
        # Sample Dirichlet noise only for legal moves
        noise = np.zeros_like(policy_probs)
        legal_noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))

        for i, action in enumerate(legal_moves):
            noise[action] = legal_noise[i]

        # Mix noise: (1-ε) * policy + ε * noise
        noised_policy = (1 - self.dirichlet_epsilon) * policy_probs + self.dirichlet_epsilon * noise

        # Renormalize
        noised_policy = noised_policy / noised_policy.sum()

        return noised_policy

    def _get_policy_target(
        self,
        root: MCTSNode,
        move_num: int,
        action_size: int
    ) -> np.ndarray:
        """Extract policy target from visit counts with temperature.

        Args:
            root: Root node after search
            move_num: Current move number
            action_size: Size of action space

        Returns:
            Policy distribution [action_size] based on visit counts
        """
        # Get visit counts for all actions
        visits = np.zeros(action_size, dtype=np.float32)

        for action, child in root.children.items():
            visits[action] = child.visit_count

        # Apply temperature
        if move_num < self.temperature_threshold:
            # High temperature: more exploration
            # visits^(1/T)
            visits = visits ** (1.0 / self.temperature)
        else:
            # Low temperature: deterministic (take argmax)
            # Set all but max to 0
            max_action = np.argmax(visits)
            visits = np.zeros_like(visits)
            visits[max_action] = 1.0

        # Normalize to probability distribution
        if visits.sum() > 0:
            policy = visits / visits.sum()
        else:
            # Should not happen, but handle gracefully
            policy = np.ones(action_size) / action_size

        return policy

    @torch.no_grad()
    def _evaluate(self, game) -> tuple:
        """Evaluate game position using HRM model.

        Args:
            game: Game instance

        Returns:
            Tuple of (policy_logits, value_logits)
            - policy_logits: [action_size]
            - value_logits: [3] for W/D/L
        """
        # Convert game state to tokens
        tokens = game.to_tokens()  # [seq_len]

        # Add batch dimension and move to device
        tokens = tokens.unsqueeze(0).to(self.device)  # [1, seq_len]

        # Get model prediction (use config-specified max_segments instead of hardcoded 10)
        policy_logits, value_logits, _ = self.model.predict(
            tokens, use_act=True, max_segments=self.max_segments_inference
        )

        # Remove batch dimension
        policy_logits = policy_logits.squeeze(0)  # [action_size]
        value_logits = value_logits.squeeze(0)    # [3]

        return policy_logits, value_logits
