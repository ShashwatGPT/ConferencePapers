# Save current Microsoft identity
ORIG_NAME=$(git config --global user.name)
ORIG_EMAIL=$(git config --global user.email)

# Set GitHub identity for push
git config --global credential.credentialStore cache
git config --global credential.guiPrompt false
git config --global user.name "ShashwatGPT"
git config --global user.email "guptashashwatme@gmail.com"

# Run the push script
# echo "# ConferencePapers" >> README.md
# git init
# git add README.md
# git commit -m "first commit"
# git branch -M main
# git remote add origin https://github.com/ShashwatGPT/ConferencePapers.git
# git push -u origin main
git add -A
git commit -m "Updates"
git push -u origin main

# Revert to Microsoft identity
git config --global user.name "${ORIG_NAME:-Shashwat}"
git config --global user.email "${ORIG_EMAIL:-t-shashgupta@microsoft.com}"

echo "Reverted to:"
git config --global user.name
git config --global user.email