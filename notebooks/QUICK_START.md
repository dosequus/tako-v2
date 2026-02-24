# 🚀 Quick Start: Private Repo in Google Colab

**Goal:** Run your private `tako-v2` notebooks in Google Colab

**Note:** This is for **organization repositories**. Personal repos? See `PRIVATE_REPO_SETUP.md`

---

## 🏢 Organization Repo? (Important!)

**Is your repo under an organization?**
- ✅ URL looks like: `github.com/YOUR_ORG/tako-v2`
- ✅ You're part of a team/organization

**Additional step required:**
- Token needs **`read:org`** scope (not just `repo`)
- May need **SSO authorization** for the org

**See full guide:** `ORG_REPO_SETUP.md` for organization-specific instructions.

---

## 5-Minute Setup (One Time Only)

### Step 1: Create GitHub Token (2 min)

1. **Visit:** https://github.com/settings/tokens
2. **Click:** "Generate new token (classic)"
3. **Configure:**
   - Name: `Colab Tako Access`
   - Expiration: 90 days (or No expiration)
   - ✅ Check **`repo`** (Full control of private repositories)
   - ✅ Check **`read:org`** (Read org membership - **REQUIRED for org repos**)
4. **Copy token:** `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - ⚠️ **Save it now!** You can't see it again.

5. **[ORG REPOS ONLY] Authorize for organization:**
   - After creating token, you may see "Configure SSO"
   - Click "Configure SSO" → Click "Authorize" for your org
   - **⚠️ Skip this if your org doesn't use SSO**

---

### Step 2: Add Token to Colab (1 min)

1. **Open any notebook** in Google Colab
2. **Click 🔑 icon** in left sidebar (Secrets)
3. **Add new secret:**
   - Name: `GITHUB_TOKEN`
   - Value: Paste your `ghp_...` token
   - ✅ Toggle ON "Notebook access"
4. **Done!** ✅

---

### Step 3: Run Notebooks (2 min)

**That's it!** The notebooks will now automatically:
- ✅ Read token from Colab Secrets
- ✅ Clone your private repo
- ✅ Install dependencies
- ✅ Start training

Just **run the cells** - no code changes needed!

---

## What You'll See

### ✅ Success (Token configured):
```
✅ Private repository cloned successfully
✅ Dependencies installed
```

### ❌ Error (Token missing):
```
❌ ERROR: Could not access GITHUB_TOKEN from Colab Secrets

This is a PRIVATE repository. Please set up authentication:
1. Create GitHub Personal Access Token...
2. Add token to Colab Secrets...
```
→ Follow the instructions to add your token

---

## Visual Guide

```
GitHub Token Creation:
┌─────────────────────────────────────────┐
│ https://github.com/settings/tokens      │
│                                         │
│ Generate new token (classic)            │
│ ├─ Name: Colab Tako Access              │
│ ├─ Expiration: 90 days                  │
│ └─ Scopes: ✅ repo                      │
│                                         │
│ [Generate token]                        │
│                                         │
│ ghp_xxxxxxxxxxxx... ← COPY THIS!        │
└─────────────────────────────────────────┘
```

```
Colab Secrets Setup:
┌─────────────────────────────────────────┐
│ Google Colab Notebook                   │
│                                         │
│ [🔑] ← Click Secrets icon               │
│  │                                      │
│  └─ Add new secret:                     │
│     ├─ Name:  GITHUB_TOKEN              │
│     ├─ Value: ghp_xxxxxxxxxxxx...       │
│     └─ [✅] Notebook access             │
│                                         │
│ [Save]                                  │
└─────────────────────────────────────────┘
```

---

## Security Notes

✅ **Safe to share notebooks** - Token is never exposed
✅ **Token stored securely** - Only you can access it
✅ **Easy to revoke** - Delete token on GitHub anytime
✅ **Works across all notebooks** - Set once, use everywhere

---

## Troubleshooting

### "Could not access GITHUB_TOKEN"
→ Add token to Colab Secrets (see Step 2)

### "Authentication failed"
→ Token expired or revoked - generate new token

### "Repository not found"
→ Verify repo name: `zfdupont/tako-v2`

---

## Next Steps

Once token is set up:

1. **Open:** `01_train_tictactoe.ipynb`
2. **Enable GPU:** Runtime → Change runtime type → GPU (T4)
3. **Run all cells**
4. **Watch it train!** 🎉

Expected results (T4 GPU):
- ~360,000 games/hour
- Converges in ~20 minutes
- 90%+ win rate vs random

---

## Full Documentation

- **Detailed guide:** `PRIVATE_REPO_SETUP.md`
- **Alternative methods:** SSH keys, Drive clone
- **Security best practices**
- **Troubleshooting**

---

**That's it! You're ready to use your private repo in Colab.** 🚀

*Questions? See `PRIVATE_REPO_SETUP.md` for detailed documentation.*
