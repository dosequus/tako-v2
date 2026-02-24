# Troubleshooting: Common Colab Clone Errors

## Error: "could not read Username for 'https://github.com'"

**Full error:**
```
fatal: could not read Username for 'https://github.com': No such device or address
```

### What This Means

Git is trying to prompt you for credentials, but:
- Colab notebooks can't show interactive prompts
- The authentication token isn't being passed to git
- Git falls back to asking for username/password (which fails)

### Root Cause

The token isn't being included in the git URL correctly.

---

## ✅ Solution 1: Use subprocess (Recommended)

**The fix is already implemented in the updated notebook!**

The issue was using:
```python
# DON'T DO THIS - doesn't work in Colab
!git clone {repo_url}
```

Instead, use:
```python
# DO THIS - works correctly
subprocess.run(['git', 'clone', repo_url, REPO_NAME])
```

**Why it works:**
- `subprocess.run()` passes the URL directly to git
- No shell interpolation issues
- Token is properly included in the authentication

---

## ✅ Solution 2: Verify Your Setup

### Step 1: Check ORG_NAME is set

In the notebook clone cell, verify:
```python
ORG_NAME = "YOUR_ACTUAL_ORG_NAME"  # ← Must be updated!
REPO_NAME = "tako-v2"
```

**Find your org name:**
- Look at your repo URL: `github.com/YOUR_ORG/tako-v2`
- `YOUR_ORG` is your organization name

### Step 2: Verify Token in Colab Secrets

1. Click 🔑 (Secrets icon) in Colab sidebar
2. Check `GITHUB_TOKEN` exists
3. Check "Notebook access" is **enabled**
4. Token should start with `ghp_`

### Step 3: Verify Token Scopes

Token must have:
- ✅ `repo` (Full control of private repositories)
- ✅ `read:org` (Read org membership)

**Check at:** https://github.com/settings/tokens

### Step 4: Verify Organization Authorization

**If your org uses SSO:**
1. Go to: https://github.com/settings/tokens
2. Find your token
3. Look for "Configure SSO" button
4. Click it
5. Click "Authorize" next to your organization

**This is the #1 cause of "Repository not found" errors!**

---

## 🔍 How to Debug

### Test 1: Verify token is accessible

Add a test cell:
```python
from google.colab import userdata

try:
    token = userdata.get('GITHUB_TOKEN')
    print(f"✅ Token found, starts with: {token[:7]}...")
    print(f"✅ Token length: {len(token)} characters")
except:
    print("❌ No token found in Colab Secrets")
```

Expected output:
```
✅ Token found, starts with: ghp_xxx...
✅ Token length: 40 characters
```

### Test 2: Verify org access with curl

Add a test cell:
```python
import subprocess
from google.colab import userdata

token = userdata.get('GITHUB_TOKEN')
ORG_NAME = "YOUR_ORG_NAME"  # Update this
REPO_NAME = "tako-v2"

# Test API access
result = subprocess.run(
    ['curl', '-H', f'Authorization: token {token}',
     f'https://api.github.com/repos/{ORG_NAME}/{REPO_NAME}'],
    capture_output=True,
    text=True
)

print(result.stdout)
```

**Expected:** Repo details (JSON)
**If error:** Shows why access is failing

### Test 3: Manual clone test

```python
import subprocess
from google.colab import userdata

token = userdata.get('GITHUB_TOKEN')
ORG_NAME = "YOUR_ORG_NAME"
REPO_NAME = "tako-v2"

url = f"https://{token}@github.com/{ORG_NAME}/{REPO_NAME}.git"

print(f"Cloning: {ORG_NAME}/{REPO_NAME}")
result = subprocess.run(
    ['git', 'clone', url, REPO_NAME],
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("✅ Clone successful!")
else:
    print("❌ Clone failed:")
    print(result.stderr)
```

---

## 📋 Common Error Messages & Fixes

### Error: "Repository not found" (404)

**Cause:** Token not authorized for organization

**Fix:**
1. https://github.com/settings/tokens
2. Find your token
3. "Configure SSO" → "Authorize" for org
4. Re-run notebook

---

### Error: "Authentication failed"

**Causes:**
- Token expired
- Token missing scopes
- Wrong token

**Fix:**
1. Create new token with `repo` + `read:org`
2. Update Colab Secret
3. Re-run notebook

---

### Error: "Resource protected by organization SAML"

**Cause:** Org uses SAML SSO, token not authorized

**Fix:**
1. https://github.com/settings/tokens
2. "Enable SSO" or "Configure SSO"
3. Authorize for organization
4. Re-run notebook

---

## ✅ Checklist: Everything Working

If your setup is correct, you should see:

**In Colab Secrets:**
```
🔑 GITHUB_TOKEN [✅ enabled for notebook]
```

**In notebook output:**
```
Attempting to clone: YOUR_ORG/tako-v2
================================================================================
✅ Retrieved GITHUB_TOKEN from Colab Secrets
🔄 Cloning repository...
✅ Repository cloned successfully
✅ Token removed from git config
✅ Setup complete!
```

**In file browser:**
```
📁 tako-v2/
  ├── model/
  ├── games/
  ├── scripts/
  └── ...
```

---

## 🆘 Still Not Working?

### Double-check everything:

1. **Token scopes:**
   - [ ] Has `repo` scope ✅
   - [ ] Has `read:org` scope ✅

2. **Organization authorization:**
   - [ ] Token authorized for org (if SSO) ✅

3. **Colab Secrets:**
   - [ ] `GITHUB_TOKEN` exists ✅
   - [ ] Notebook access enabled ✅
   - [ ] Token starts with `ghp_` ✅

4. **Notebook configuration:**
   - [ ] `ORG_NAME` updated ✅
   - [ ] `REPO_NAME` correct ✅

5. **Access verification:**
   - [ ] Can see repo on github.com ✅
   - [ ] You're logged in to GitHub ✅
   - [ ] You're a member of the org ✅

### If ALL checked and still fails:

**Contact your organization admin:**
- "Do I have access to `ORG_NAME/tako-v2`?"
- "Are there IP restrictions?" (may block Colab)
- "Is 2FA required?" (ensure enabled on your account)
- "Are there special SSO settings?"

---

## 🎯 Quick Fix Summary

**Most common issue:** Token not authorized for organization

**Quick fix:**
1. Visit: https://github.com/settings/tokens
2. Find your token
3. Click "Configure SSO"
4. Click "Authorize" next to your organization
5. Re-run notebook

**Takes 30 seconds to fix!**

---

## Error: "RuntimeError: No CUDA GPUs are available" (Ray Workers)

**Full error:**
```
ray.exceptions.ActorDiedError: The actor died because of an error raised in its creation task
RuntimeError: No CUDA GPUs are available
```

### What This Means

Ray workers can't access the GPU, even though the main process detects CUDA.

### Root Cause

Ray needs explicit resource allocation. When you create Ray actors without requesting GPU resources, Ray doesn't give them GPU access.

---

## ✅ Solution: Fractional GPU Allocation

**The fix is already implemented in the updated training script!**

The issue was:
```python
# DON'T DO THIS - workers can't access GPU
worker = SelfPlayWorker.remote(...)
```

Instead, use:
```python
# DO THIS - request fractional GPU for each worker
gpu_fraction = num_gpus / num_workers  # e.g., 1 GPU / 8 workers = 0.125 per worker
worker = SelfPlayWorker.options(num_gpus=gpu_fraction).remote(...)
```

**Why it works:**
- Ray allows fractional GPU allocation (e.g., 0.125 GPU per worker)
- Multiple workers can share a single GPU
- In Colab with 1 GPU + 8 workers, each gets 1/8th of GPU resources

**Expected output:**
```
[Train] Detected 1 CUDA GPU(s)
[Train] Workers created on cuda device(s)
[Train]   Each worker allocated 0.125 GPU (8 workers sharing 1 GPU(s))
```

---

## 🔍 Verification: Check GPU Allocation

After starting training, verify Ray workers have GPU access:

```python
# Check Ray actor resource allocation
import ray
ray.available_resources()  # Should show fractional GPU usage
```

Expected:
```
{'CPU': 8.0, 'GPU': 1.0, ...}  # Before workers start
{'CPU': 8.0, 'GPU': 0.0, ...}  # After workers claim GPU fractions
```

---

## 📋 Common Colab GPU Issues

### Issue 1: "No CUDA GPUs are available"

**Cause:** GPU not enabled in Colab runtime

**Fix:**
1. Runtime → Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: T4 (free), V100/A100 (Colab Pro)
4. Save → Restart session

---

### Issue 2: GPU detected but training still slow

**Cause:** Workers might be on CPU despite GPU detection

**Check logs for:**
```
[Train] Workers created on cpu device(s)  # ← Bad!
```

Should see:
```
[Train] Workers created on cuda device(s)  # ← Good!
[Train]   Each worker allocated 0.125 GPU...
```

**Fix:** Update to latest `scripts/train.py` with fractional GPU support

---

### Issue 3: "CUDA out of memory"

**Causes:**
- Too many workers for GPU memory
- Batch size too large
- Model too large

**Solutions:**

**Option 1: Reduce num_workers**
```yaml
# config/tictactoe.yaml
selfplay:
  num_workers: 4  # Was: 8
```

**Option 2: Reduce batch_size**
```yaml
training:
  batch_size: 256  # Was: 512
```

**Option 3: Use CPU workers + GPU learner**

In training script, force CPU for workers:
```python
# In train.py, override device detection
device_type = 'cpu'  # Force CPU for workers
```

This uses CPU for game generation, GPU only for neural network training.

---

### Issue 4: Ray dashboard not accessible

**Symptom:** Can't view Ray dashboard at `localhost:8265`

**Solution:** Use ngrok tunnel in Colab

```python
# Install ngrok
!pip install pyngrok

# Create tunnel
from pyngrok import ngrok
public_url = ngrok.connect(8265)
print(f"Ray Dashboard: {public_url}")
```

Then visit the ngrok URL to see Ray dashboard.

---

## 📚 Related Documentation

- **Organization setup:** `ORG_REPO_GUIDE.md`
- **Detailed guide:** `notebooks/ORG_REPO_SETUP.md`
- **Quick start:** `notebooks/QUICK_START.md`

---

**Still stuck?** Create an issue with:
- Exact error message
- Token scopes (screenshot with token value hidden)
- Organization name (if not private)
- Steps you've tried

We'll help debug! 🔧
