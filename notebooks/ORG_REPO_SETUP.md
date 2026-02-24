# Setting Up Private Organization Repo in Google Colab

**Special considerations for organization repositories (not personal repos)**

---

## 🏢 Organization Repo vs Personal Repo

Your repo: `https://github.com/YOUR_ORG/tako-v2` (organization-owned)

**Key differences:**
- ✅ May require SSO authorization
- ✅ May need org-specific permissions
- ✅ Token must be authorized for the organization

---

## ✅ Setup for Organization Repository

### Step 1: Create GitHub Personal Access Token

1. **Go to GitHub Settings:**
   - Visit: https://github.com/settings/tokens
   - Or: Your profile → Settings → Developer settings → Personal access tokens → Tokens (classic)

2. **Generate new token (classic):**
   - Click "Generate new token (classic)"
   - **Name:** `Colab Tako Access`
   - **Expiration:** 90 days (or No expiration)

3. **Select scopes:**
   - ✅ **`repo`** (Full control of private repositories)
   - ✅ **`read:org`** (Read org and team membership - **REQUIRED for org repos**)

4. **Generate and copy:**
   - Click "Generate token"
   - **⚠️ COPY NOW!** You won't see it again
   - Token format: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

---

### Step 2: Authorize Token for Organization (IMPORTANT!)

**If your organization uses SAML SSO or has additional security:**

1. **After creating the token, you'll see:**
   ```
   Configure SSO ▼
   ```

2. **Click "Configure SSO"** next to your organization name

3. **Click "Authorize"** for your organization

4. **Confirm authorization**

**⚠️ Without this step, the token won't work for org repos!**

---

### Step 3: Verify Organization Access

**Test your token has access:**

```bash
# On your local machine (optional verification)
curl -H "Authorization: token ghp_YOUR_TOKEN" \
     https://api.github.com/repos/YOUR_ORG/tako-v2

# Should return repo details, not 404
```

---

### Step 4: Add Token to Colab Secrets

**Same as personal repos:**

1. **Open any Colab notebook**

2. **Click 🔑 (Secrets) in left sidebar**

3. **Add new secret:**
   - Name: `GITHUB_TOKEN`
   - Value: `ghp_your_token_here`
   - Toggle: ✅ Enable "Notebook access"

4. **Done!**

---

## 🔧 Updated Notebook Clone Cell

**For organization repos, update the repo URL in notebooks:**

```python
# Clone PRIVATE ORGANIZATION repository using GitHub token
import os
from google.colab import userdata

# Get token from Colab Secrets
try:
    github_token = userdata.get('GITHUB_TOKEN')

    # IMPORTANT: Update with your organization name and repo name
    ORG_NAME = "YOUR_ORG_NAME"      # Replace with your org
    REPO_NAME = "tako-v2"

    repo_url = f"https://{github_token}@github.com/{ORG_NAME}/{REPO_NAME}.git"

    if not os.path.exists(REPO_NAME):
        print(f"Cloning {ORG_NAME}/{REPO_NAME}...")
        !git clone {repo_url} {REPO_NAME}

        # Remove token from git config for security
        !cd {REPO_NAME} && git remote set-url origin https://github.com/{ORG_NAME}/{REPO_NAME}.git

        print(f"✅ Private organization repository cloned successfully")
    else:
        print(f"✅ Repository already exists")

except Exception as e:
    print("\n" + "="*80)
    print("❌ ERROR: Could not access GITHUB_TOKEN from Colab Secrets")
    print("="*80)
    print("\nThis is a PRIVATE ORGANIZATION repository.")
    print("\n📋 Setup Checklist:")
    print("\n1. Create GitHub Personal Access Token:")
    print("   → https://github.com/settings/tokens")
    print("   → Generate new token (classic)")
    print("   → Check 'repo' AND 'read:org' scopes")
    print("   → Copy the token (starts with ghp_)")
    print("\n2. Authorize token for your organization:")
    print("   → After creating token, click 'Configure SSO'")
    print("   → Click 'Authorize' next to your organization")
    print("   → Confirm authorization")
    print("\n3. Add token to Colab Secrets:")
    print("   → Click the 🔑 key icon in the left sidebar")
    print("   → Add secret: Name='GITHUB_TOKEN', Value='ghp_...'")
    print("   → Enable 'Notebook access'")
    print("\n4. Re-run this cell")
    print("\n⚠️  Make sure you authorized the token for the organization!")
    print("="*80 + "\n")
    raise e

%cd {REPO_NAME}
```

---

## 🔍 Common Issues with Organization Repos

### Issue 1: "Repository not found" (404)

**Causes:**
- Token not authorized for organization (most common)
- Wrong org name or repo name
- No access to the repo

**Solutions:**
1. **Authorize token for org:**
   - Go to https://github.com/settings/tokens
   - Find your token
   - Click "Configure SSO"
   - Click "Authorize" next to your org

2. **Verify org/repo names:**
   - Check URL: `https://github.com/YOUR_ORG/tako-v2`
   - Update `ORG_NAME` in notebook

3. **Check repo access:**
   - Visit repo on GitHub
   - Confirm you can see it while logged in

---

### Issue 2: "Resource protected by organization SAML enforcement"

**Cause:** Token not authorized for SSO-protected org

**Solution:**
1. Go to: https://github.com/settings/tokens
2. Find your token
3. Click "Configure SSO" or "Enable SSO"
4. Authorize for your organization
5. Re-run notebook

---

### Issue 3: "Bad credentials"

**Causes:**
- Token expired
- Token deleted
- Wrong token pasted

**Solutions:**
- Generate new token
- Update Colab Secret with new token
- Verify token starts with `ghp_`

---

## 🔒 Required Token Scopes for Org Repos

**Minimum required:**
```
✅ repo                    (Full control of private repositories)
✅ read:org                (Read org and team membership, team discussions)
```

**Optional but recommended:**
```
✅ workflow                (If you use GitHub Actions)
```

**NOT needed:**
```
❌ admin:org               (Too much permission!)
❌ delete_repo             (Dangerous!)
❌ admin:public_key        (Not needed)
```

**Use minimal permissions for security!**

---

## 🏢 Organization-Specific Settings

### Check Organization Settings

**As an org member, verify:**

1. **You have repo access:**
   - Visit `https://github.com/YOUR_ORG/tako-v2`
   - Can you see the code?

2. **Check your org role:**
   - Organization → People
   - Your role: Member, Admin, or Owner?

3. **Check repo permissions:**
   - Repo → Settings → Manage access
   - Do you have Read/Write/Admin?

**If not, ask org admin to grant access!**

---

## 📋 Complete Setup Checklist for Org Repos

### ☑️ GitHub Setup
- [ ] Create Personal Access Token (classic)
- [ ] Select scopes: `repo` + `read:org`
- [ ] **Authorize token for organization** (SSO if applicable)
- [ ] Copy token (starts with `ghp_`)
- [ ] Verify you can access the repo on GitHub

### ☑️ Colab Setup
- [ ] Open Colab notebook
- [ ] Click 🔑 Secrets icon
- [ ] Add: `GITHUB_TOKEN` = your token
- [ ] Enable: Notebook access
- [ ] Update: `ORG_NAME` in clone cell

### ☑️ Test
- [ ] Run clone cell
- [ ] See: "✅ Private organization repository cloned"
- [ ] No errors!

---

## 🎯 Quick Reference: Org vs Personal Repos

| Feature | Personal Repo | Organization Repo |
|---------|---------------|-------------------|
| **Token scopes** | `repo` | `repo` + `read:org` |
| **SSO authorization** | Not needed | **May be required** |
| **Repo URL** | `github.com/username/repo` | `github.com/org/repo` |
| **Access control** | Just you | Org members + teams |
| **Setup complexity** | Simple | +1 step (SSO auth) |

---

## 🔐 Security Best Practices for Org Repos

### ✅ DO:
- ✅ Use minimal scopes (`repo` + `read:org` only)
- ✅ Set token expiration (90 days recommended)
- ✅ Authorize token for specific org only
- ✅ Delete token when no longer needed
- ✅ Use different tokens for different purposes

### ❌ DON'T:
- ❌ Use tokens with `admin:org` scope (too powerful!)
- ❌ Share tokens with team members (each person creates their own)
- ❌ Use no-expiration tokens for org repos (security risk)
- ❌ Hardcode tokens in notebooks (use Colab Secrets!)

---

## 🆘 Troubleshooting Steps

### If clone fails:

1. **Verify token scopes:**
   ```
   Visit: https://github.com/settings/tokens
   Check: repo ✅, read:org ✅
   ```

2. **Verify org authorization:**
   ```
   On token page: Configure SSO → Authorize
   If no SSO: Skip this
   ```

3. **Test token manually:**
   ```bash
   curl -H "Authorization: token YOUR_TOKEN" \
        https://api.github.com/repos/YOUR_ORG/tako-v2

   # Should return: repo details (not 404)
   ```

4. **Verify repo access:**
   ```
   Visit: https://github.com/YOUR_ORG/tako-v2
   Can you see the code while logged in?
   ```

5. **Check Colab Secret:**
   ```
   🔑 Secrets → GITHUB_TOKEN exists?
   Notebook access enabled?
   ```

---

## 📞 Getting Help

### If nothing works:

**Check with org admin:**
- Do I have access to `YOUR_ORG/tako-v2`?
- Does the org require specific security settings?
- Are there IP restrictions or 2FA requirements?

**Common org restrictions:**
- IP allowlists (may block Colab)
- Required 2FA (ensure your account has it)
- SSO requirements (ensure token is authorized)

---

## ✅ Example: Complete Setup for Org Repo

**Scenario:** Cloning `acme-corp/tako-v2` (organization repo)

### Step-by-step:

1. **Create token:**
   - https://github.com/settings/tokens
   - Scopes: `repo`, `read:org`
   - Copy: `ghp_abc123xyz...`

2. **Authorize for org (if SSO):**
   - Click "Configure SSO"
   - Authorize for `acme-corp`

3. **Add to Colab:**
   - 🔑 → `GITHUB_TOKEN` = `ghp_abc123xyz...`

4. **Update notebook:**
   ```python
   ORG_NAME = "acme-corp"
   REPO_NAME = "tako-v2"
   ```

5. **Run:**
   ```
   ✅ Private organization repository cloned successfully
   ```

**Done!** 🎉

---

## 📚 Additional Resources

- **GitHub Docs:** [Authorizing PAT with SSO](https://docs.github.com/en/authentication/authenticating-with-saml-single-sign-on/authorizing-a-personal-access-token-for-use-with-saml-single-sign-on)
- **Org permissions:** [Organization roles](https://docs.github.com/en/organizations/managing-peoples-access-to-your-organization-with-roles/roles-in-an-organization)
- **Token scopes:** [GitHub token scopes](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps)

---

**Your organization repo is now ready for Colab!** 🏢✨

*For general setup (non-org), see `PRIVATE_REPO_SETUP.md`*
