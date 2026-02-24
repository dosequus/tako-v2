# Setting Up Private GitHub Repo in Google Colab

This guide shows how to securely access your private `tako-v2` repository in Google Colab.

---

## ✅ Recommended Method: GitHub Personal Access Token (PAT)

**Why this method?**
- ✅ Secure (uses Colab Secrets, never exposed in notebooks)
- ✅ Easy to set up (5 minutes)
- ✅ Works automatically across all notebooks
- ✅ No passwords in code
- ✅ Can be revoked anytime

---

## Step-by-Step Setup

### 1. Create a GitHub Personal Access Token

1. **Go to GitHub:**
   - Visit: https://github.com/settings/tokens
   - Or: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)

2. **Generate new token:**
   - Click "Generate new token (classic)"
   - Name it: `Colab Tako Access`
   - Expiration: Choose duration (90 days recommended, or "No expiration" for convenience)

3. **Select scopes:**
   - ✅ Check **`repo`** (Full control of private repositories)
   - That's it! No other scopes needed.

4. **Generate and copy:**
   - Click "Generate token"
   - **⚠️ COPY THE TOKEN NOW!** You won't see it again.
   - Format: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

---

### 2. Store Token in Colab Secrets

**New Colab Secrets Feature (Recommended):**

1. **Open any Colab notebook**

2. **Access Secrets:**
   - Click the 🔑 **key icon** in the left sidebar (Secrets)
   - Or: Tools → Secrets

3. **Add new secret:**
   - Name: `GITHUB_TOKEN`
   - Value: Paste your token (starts with `ghp_`)
   - Toggle: **Enable "Notebook access"** for notebooks that need it

4. **Done!** This token is now securely stored and never visible in your notebooks.

**Screenshot:**
```
🔑 Secrets
├─ GITHUB_TOKEN  [enabled for notebook]
│  └─ ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxx (hidden)
```

---

### 3. Update Notebook to Use Secret

**Replace the clone cell in `00_setup_and_benchmark.ipynb`:**

```python
# Clone private repository using GitHub token from Colab Secrets
import os
from google.colab import userdata

# Get token from Colab Secrets
try:
    github_token = userdata.get('GITHUB_TOKEN')

    # Clone private repo with authentication
    repo_url = f"https://{github_token}@github.com/zfdupont/tako-v2.git"

    if not os.path.exists('tako-v2'):
        !git clone {repo_url} tako-v2
        # Remove token from git config for security
        !cd tako-v2 && git remote set-url origin https://github.com/zfdupont/tako-v2.git
        print("✅ Private repository cloned successfully")
    else:
        print("✅ Repository already exists")

except Exception as e:
    print("❌ Error: Could not access GITHUB_TOKEN from Colab Secrets")
    print("   Please add your GitHub Personal Access Token to Colab Secrets:")
    print("   1. Click the 🔑 key icon in the left sidebar")
    print("   2. Add secret: Name='GITHUB_TOKEN', Value='ghp_your_token_here'")
    print("   3. Enable 'Notebook access'")
    raise e

%cd tako-v2
```

**What this does:**
- ✅ Reads token from secure Colab Secrets
- ✅ Clones your private repo
- ✅ Removes token from git config (so it's not saved in Drive)
- ✅ Shows helpful error message if token is missing

---

## Alternative Methods

### Option 2: Google Drive One-Time Clone

**Pros:** Clone once, reuse forever
**Cons:** Uses Drive storage, slower access

**Setup:**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# First time only: Clone to Drive using token
import os
from google.colab import userdata

drive_path = '/content/drive/MyDrive/tako-v2'

if not os.path.exists(drive_path):
    github_token = userdata.get('GITHUB_TOKEN')
    repo_url = f"https://{github_token}@github.com/zfdupont/tako-v2.git"
    !git clone {repo_url} {drive_path}
    print("✅ Cloned to Google Drive (one-time setup)")

# Create symlink to Drive copy
!ln -s {drive_path} /content/tako-v2
%cd /content/tako-v2

print("✅ Using repository from Google Drive")
```

**Benefits:**
- Clone once, never again
- Survives session restarts
- Can edit code and commit from Colab

---

### Option 3: SSH Keys (Advanced)

**Pros:** Most secure, no tokens
**Cons:** Complex setup, need to configure every session

<details>
<summary>Click to expand SSH setup</summary>

**One-time GitHub setup:**
1. Generate SSH key locally: `ssh-keygen -t ed25519 -C "colab@example.com"`
2. Add public key to GitHub: Settings → SSH and GPG keys
3. Store private key in Colab Secrets: Name=`SSH_PRIVATE_KEY`

**Notebook setup:**
```python
from google.colab import userdata
import os

# Setup SSH
ssh_dir = os.path.expanduser('~/.ssh')
os.makedirs(ssh_dir, exist_ok=True)

# Write private key
private_key = userdata.get('SSH_PRIVATE_KEY')
with open(f'{ssh_dir}/id_ed25519', 'w') as f:
    f.write(private_key)
os.chmod(f'{ssh_dir}/id_ed25519', 0o600)

# Add GitHub to known_hosts
!ssh-keyscan github.com >> ~/.ssh/known_hosts

# Clone using SSH
!git clone git@github.com:zfdupont/tako-v2.git
```

</details>

---

## Security Best Practices

### ✅ DO:
- ✅ Use Colab Secrets for tokens (never hardcode in notebooks)
- ✅ Set token expiration (90 days recommended)
- ✅ Use minimal scopes (only `repo` for private access)
- ✅ Revoke tokens you no longer use
- ✅ Remove tokens from git config after cloning

### ❌ DON'T:
- ❌ Hardcode tokens in notebook cells
- ❌ Share notebooks with tokens embedded
- ❌ Use your GitHub password (deprecated by GitHub)
- ❌ Give tokens more permissions than needed
- ❌ Commit notebooks with credentials to GitHub

---

## Updating Notebooks

I'll create updated versions of all notebooks with proper private repo authentication.

**Files to update:**
- `00_setup_and_benchmark.ipynb` - Use Colab Secrets
- `01_train_tictactoe.ipynb` - Use Colab Secrets
- `02_evaluate_model.ipynb` - Use Colab Secrets
- `03_interactive_play.ipynb` - Use Colab Secrets

**Changes:**
- Replace hardcoded clone with secure token method
- Add helpful error messages
- Show setup instructions if token missing

---

## Troubleshooting

### "Could not access GITHUB_TOKEN"

**Solution:**
1. Open Colab Secrets (🔑 icon in left sidebar)
2. Add secret: `GITHUB_TOKEN` = your token
3. Enable "Notebook access"
4. Re-run the cell

### "Authentication failed"

**Causes:**
- Token expired → Generate new token
- Wrong scopes → Ensure `repo` scope is checked
- Token revoked → Generate new token

**Solution:**
1. Generate new token: https://github.com/settings/tokens
2. Update Colab Secret with new token
3. Re-run clone cell

### "Repository not found"

**Causes:**
- Repo URL is wrong
- Token doesn't have access to this repo

**Solution:**
- Verify repo URL: `https://github.com/zfdupont/tako-v2.git`
- Ensure token has `repo` scope

---

## FAQ

### Q: How long does the token last?
**A:** You choose when creating it (7 days, 30 days, 90 days, or no expiration). Recommend 90 days.

### Q: Can I use the same token for multiple repos?
**A:** Yes! One token with `repo` scope works for all your private repos.

### Q: What if I share my notebook publicly?
**A:** Safe! Colab Secrets are never exposed in shared notebooks. Others will be prompted to add their own token.

### Q: Can I revoke the token?
**A:** Yes! Go to GitHub → Settings → Developer settings → Tokens → Delete. Colab loses access immediately.

### Q: Do I need to do this for every Colab session?
**A:** No! Once you add the token to Colab Secrets, it persists across all sessions.

---

## Quick Reference

**Create token:**
https://github.com/settings/tokens → Generate new token (classic) → Check `repo` → Copy token

**Add to Colab:**
🔑 Secrets icon → Add secret → Name: `GITHUB_TOKEN`, Value: your token → Enable notebook access

**Clone in notebook:**
```python
from google.colab import userdata
token = userdata.get('GITHUB_TOKEN')
!git clone https://{token}@github.com/zfdupont/tako-v2.git
```

**Done!** 🎉

---

*Last updated: 2026-02-23*
