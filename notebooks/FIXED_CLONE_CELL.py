"""
Fixed clone cell for Colab notebooks - handles authentication correctly

Replace the clone cell in your notebooks with this code.
"""

# Clone PRIVATE ORGANIZATION repository using GitHub token
import os
import subprocess
from google.colab import userdata

# Configuration - UPDATE THESE!
ORG_NAME = "YOUR_ORG_NAME"  # Replace with your organization name
REPO_NAME = "tako-v2"        # Repository name

def clone_private_repo():
    """Clone private organization repo with proper authentication."""

    # Get token from Colab Secrets
    try:
        github_token = userdata.get('GITHUB_TOKEN')
        print(f"✅ Retrieved GITHUB_TOKEN from Colab Secrets")
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR: Could not access GITHUB_TOKEN from Colab Secrets")
        print("="*80)
        print("\n🏢 This is a PRIVATE ORGANIZATION repository.")
        print("\n📋 Setup Instructions:")
        print("\n1. Create GitHub Personal Access Token:")
        print("   → https://github.com/settings/tokens")
        print("   → Generate new token (classic)")
        print("   → Check TWO scopes:")
        print("      ✅ repo (Full control of private repositories)")
        print("      ✅ read:org (Read org membership) ← REQUIRED FOR ORG REPOS!")
        print("   → Copy token: ghp_xxxxxxxxxxxx...")
        print("\n2. Authorize token for your organization:")
        print("   → After creating token, look for 'Configure SSO' button")
        print("   → Click 'Configure SSO'")
        print("   → Click 'Authorize' next to your organization")
        print("   → (Skip if you don't see 'Configure SSO')")
        print("\n3. Add token to Colab Secrets:")
        print("   → Click the 🔑 key icon in the left sidebar")
        print("   → Add secret: Name='GITHUB_TOKEN', Value='ghp_...'")
        print("   → Toggle ON 'Notebook access'")
        print("\n4. Update ORG_NAME in this cell:")
        print(f"   → Currently set to: '{ORG_NAME}'")
        print(f"   → Should be your org name from: github.com/YOUR_ORG/{REPO_NAME}")
        print("\n5. Re-run this cell")
        print("\n📚 Full guide: notebooks/ORG_REPO_SETUP.md")
        print("="*80 + "\n")
        raise e

    # Check if repo already exists
    if os.path.exists(REPO_NAME):
        print(f"✅ Repository '{REPO_NAME}' already exists")
        return True

    # Construct authenticated URL
    # Format: https://TOKEN@github.com/ORG/REPO.git
    repo_url = f"https://{github_token}@github.com/{ORG_NAME}/{REPO_NAME}.git"

    print(f"\n🔄 Cloning {ORG_NAME}/{REPO_NAME}...")

    # Clone using subprocess to avoid shell escaping issues
    try:
        result = subprocess.run(
            ['git', 'clone', repo_url, REPO_NAME],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"✅ Repository cloned successfully")

            # Remove token from git config for security
            subprocess.run(
                ['git', '-C', REPO_NAME, 'remote', 'set-url', 'origin',
                 f'https://github.com/{ORG_NAME}/{REPO_NAME}.git'],
                capture_output=True
            )
            print(f"✅ Token removed from git config (security)")

            return True
        else:
            # Clone failed - show helpful error message
            print("\n" + "="*80)
            print("❌ ERROR: Git clone failed")
            print("="*80)

            stderr = result.stderr.lower()

            if 'repository not found' in stderr or '404' in stderr:
                print("\n🔍 Error: Repository not found (404)")
                print("\nMost common causes:")
                print("\n1. Token not authorized for organization:")
                print("   → https://github.com/settings/tokens")
                print("   → Find your token")
                print("   → Click 'Configure SSO' (if visible)")
                print("   → Click 'Authorize' next to your organization")
                print("   → Re-run this cell")
                print("\n2. Wrong organization or repo name:")
                print(f"   → Currently trying: {ORG_NAME}/{REPO_NAME}")
                print(f"   → Check your repo URL on GitHub")
                print(f"   → Update ORG_NAME and REPO_NAME if incorrect")
                print("\n3. No access to the repository:")
                print(f"   → Visit: https://github.com/{ORG_NAME}/{REPO_NAME}")
                print("   → Can you see it while logged in to GitHub?")
                print("   → If not, ask org admin for access")

            elif 'authentication failed' in stderr or 'invalid credentials' in stderr:
                print("\n🔑 Error: Authentication failed")
                print("\nPossible causes:")
                print("\n1. Token expired:")
                print("   → Generate new token at: https://github.com/settings/tokens")
                print("   → Update GITHUB_TOKEN in Colab Secrets")
                print("\n2. Token missing required scopes:")
                print("   → Token needs: repo ✅ AND read:org ✅")
                print("   → Re-create token with both scopes")
                print("\n3. Wrong token in Colab Secrets:")
                print("   → Verify token starts with 'ghp_'")
                print("   → Re-check Colab Secrets (🔑 icon)")

            else:
                print(f"\n📋 Git error output:")
                print(result.stderr)
                print("\n💡 Try:")
                print("1. Verify token has 'repo' AND 'read:org' scopes")
                print("2. Ensure token is authorized for your organization (SSO)")
                print("3. Check organization and repo names are correct")

            print("\n📚 Full troubleshooting guide: notebooks/ORG_REPO_SETUP.md")
            print("="*80 + "\n")
            return False

    except subprocess.TimeoutExpired:
        print("❌ ERROR: Clone timed out (>60 seconds)")
        print("   → Check your internet connection")
        print("   → Try again")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error during clone: {e}")
        return False

# Run the clone
if clone_private_repo():
    # Change to repo directory
    os.chdir(REPO_NAME)
    print(f"\n📂 Changed to directory: {os.getcwd()}")

    # Install dependencies
    print("\n📦 Installing dependencies with uv...")
    result = subprocess.run(
        [os.path.expanduser('~/.cargo/bin/uv'), 'sync'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Dependencies installed")
    else:
        print("⚠️  Warning: Dependency installation had issues")
        print(result.stderr)

    print("\n" + "="*80)
    print("✅ Setup complete! Repository ready to use.")
    print("="*80)
else:
    print("\n❌ Setup failed. Please fix the errors above and try again.")
