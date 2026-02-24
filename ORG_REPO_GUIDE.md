# Organization Repository - Colab Setup Guide

**Your repo is under a GitHub organization, not a personal account.**

This requires **one extra step** compared to personal repos.

---

## 🏢 What's Different for Organization Repos?

| Aspect | Personal Repo | Organization Repo |
|--------|---------------|-------------------|
| **Token scopes** | `repo` | `repo` + `read:org` ✅ |
| **SSO authorization** | Not needed | **May be required** ✅ |
| **Setup time** | 5 minutes | 6 minutes (+1 step) |

**The extra minute:** Authorizing token for your organization

---

## ✅ Complete Setup (6 Minutes)

### Step 1: Create GitHub Token (3 min)

**Visit:** https://github.com/settings/tokens

**Create token with these settings:**
```
Name: Colab Tako Access
Expiration: 90 days (recommended)

Scopes:
✅ repo                    (Full control of private repositories)
✅ read:org                (Read org membership) ← REQUIRED FOR ORG REPOS!

Click: Generate token
Copy: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

### Step 2: Authorize Token for Organization (1 min) ⚠️

**This is the critical step for organization repos!**

**After creating the token:**
1. You'll see your new token on the page
2. Look for "Configure SSO" dropdown next to it
3. Click "Configure SSO"
4. Find your organization in the list
5. Click "Authorize" next to your organization name
6. Confirm authorization

**Without this step, you'll get "Repository not found" errors!**

**Note:** If you don't see "Configure SSO", your org doesn't use it (skip this step)

---

### Step 3: Add Token to Colab Secrets (1 min)

**In any Colab notebook:**
1. Click 🔑 (Secrets icon) in left sidebar
2. Add new secret:
   - Name: `GITHUB_TOKEN`
   - Value: Paste your `ghp_...` token
   - Toggle: ✅ Enable "Notebook access"
3. Save

---

### Step 4: Update Notebook with Org Name (1 min)

**Edit the clone cell in your notebooks:**

Replace:
```python
repo_url = f"https://{github_token}@github.com/zfdupont/tako-v2.git"
```

With:
```python
ORG_NAME = "YOUR_ORG_NAME"  # Replace with actual org name
REPO_NAME = "tako-v2"
repo_url = f"https://{github_token}@github.com/{ORG_NAME}/{REPO_NAME}.git"
```

**Find your org name:** Look at your repo URL: `github.com/YOUR_ORG_NAME/tako-v2`

---

### Step 5: Run Notebook (Done!)

**Just run the notebook cells!**

Expected output:
```
✅ Private organization repository cloned successfully
✅ Dependencies installed
```

---

## 🔍 Troubleshooting Organization Repos

### Error: "Repository not found" (404)

**Most common cause:** Token not authorized for organization

**Fix:**
1. Go to https://github.com/settings/tokens
2. Find your token
3. Click "Configure SSO" (if visible)
4. Click "Authorize" next to your organization
5. Re-run notebook

---

### Error: "Resource protected by organization SAML enforcement"

**Cause:** Organization uses SAML SSO, token not authorized

**Fix:**
1. Visit https://github.com/settings/tokens
2. Find your token
3. Click "Enable SSO" or "Configure SSO"
4. Authorize for your organization
5. Re-run notebook

---

### Error: "Bad credentials"

**Possible causes:**
- Token expired → Create new token
- Token deleted → Create new token
- Wrong scope → Add `read:org` scope
- Wrong token → Verify token in Colab Secrets

---

## 📋 Pre-Flight Checklist

**Before creating token:**
- [ ] I can access the repo on GitHub (while logged in)
- [ ] I know my organization name
- [ ] I know if my org uses SSO (check for "Configure SSO" option)

**After creating token:**
- [ ] Token has `repo` scope ✅
- [ ] Token has `read:org` scope ✅
- [ ] Token is authorized for organization (if SSO enabled) ✅
- [ ] Token copied to clipboard ✅

**In Colab:**
- [ ] Token added to Secrets as `GITHUB_TOKEN` ✅
- [ ] "Notebook access" toggle enabled ✅
- [ ] Org name updated in clone cell ✅

**Test:**
- [ ] Run notebook clone cell
- [ ] See "✅ Private organization repository cloned" ✅
- [ ] No errors ✅

---

## 🎯 Visual Guide: Token Scopes

**For organization repos, you MUST check both:**

```
Personal Access Token Scopes:

✅ repo                              ← Check this
   ├─ repo:status
   ├─ repo_deployment
   ├─ public_repo
   └─ repo:invite

✅ read:org                          ← AND check this!
   └─ Read org and team membership
```

**Common mistake:** Only checking `repo`, forgetting `read:org`

---

## 🔐 Why Does Organization Access Work This Way?

**Security reasons:**

1. **Organizations can have private data**
   - `repo` scope alone isn't enough to prove org membership
   - `read:org` verifies you're authorized to access org resources

2. **SSO adds another layer**
   - Some orgs require SAML SSO authentication
   - Token must be explicitly authorized per-organization

3. **Principle of least privilege**
   - Each scope grants only necessary permissions
   - Multiple scopes = multiple verification layers

**This is GitHub's security by design!**

---

## 💡 Quick Reference

### Create Token
```
https://github.com/settings/tokens
→ New token (classic)
→ Scopes: repo ✅, read:org ✅
→ Generate → Copy ghp_...
```

### Authorize for Org (if SSO)
```
Same page → Configure SSO
→ Find your org → Authorize
```

### Add to Colab
```
Colab → 🔑 → GITHUB_TOKEN = ghp_...
```

### Update Notebook
```python
ORG_NAME = "your-org-name"
REPO_NAME = "tako-v2"
```

### Run
```
✅ Cloned successfully!
```

---

## 📚 Additional Documentation

- **Detailed guide:** `notebooks/ORG_REPO_SETUP.md`
- **General private repo:** `notebooks/PRIVATE_REPO_SETUP.md`
- **Quick start:** `notebooks/QUICK_START.md`

---

## 🆘 Still Having Issues?

### Check with your organization admin:

1. **Do I have access to the repo?**
   - Visit `github.com/YOUR_ORG/tako-v2`
   - Can you see the code?

2. **Does the org have special requirements?**
   - IP allowlists (may block Colab)
   - 2FA requirements
   - Custom SSO settings

3. **What are my permissions?**
   - Repo → Settings → Manage access
   - Your role: Read, Write, or Admin?

### If admin says you have access but it still fails:

1. **Re-create token** with both scopes
2. **Re-authorize** for organization
3. **Verify** token in Colab Secrets
4. **Try** in a fresh notebook

---

## ✅ Success Indicators

**You did it right when:**

1. **Token page shows:**
   ```
   ✅ Scopes: repo, read:org
   ✅ SSO: Authorized (if applicable)
   ```

2. **Colab Secrets shows:**
   ```
   🔑 GITHUB_TOKEN [enabled for notebook]
   ```

3. **Notebook outputs:**
   ```
   ✅ Private organization repository cloned successfully
   ```

4. **You can run:**
   ```
   ls tako-v2/
   → model/ games/ scripts/ ...
   ```

---

**Your organization repo is now Colab-ready!** 🏢✨

*Questions? See `notebooks/ORG_REPO_SETUP.md` for comprehensive troubleshooting.*
