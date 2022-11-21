# Colosseum Survival - Game Playing Agent
- Eric
- Mike

### Initial Setup on your machine
1. Make sure you have Git installed on your machine.
2. Create a new folder to contain this project.
3. Inside the folder, run the following:
```
git clone https://github.com/ericjardon/game-playing-agent.git
```

4. Then, follow the instructions in this directory's `Instructions.md` file.

## Contributing to the project
1. Anytime you wish to make changes to the code, it is safe practice to "pull" from main (i.e. fetch the latest changes).
3. **Creating your branch:** It is good practice to use your own working branch (a copy of code in which you are currently working on, making updates, etc). To create your own branch from the main branch:
```
git status                      # to display current branch information. Should be in 'main'
git pull origin main            # to fetch latest changes from the remote (code in Github)
git checkout -b "my_branch"     # create a new branch with "my_branch" name
```
**NOTE:** for simple projects and features, it is good enough for 1-2 people to work on the main branch as long as they always fetch latest from the repository. 

4. **Pushing content to your branch:** Once you have made progress on your branch and wish to save your changes, you should "commit" and "push" to the repository. You can push to any branch, but it is safe practice to push to your own working branch (see above).
```
git add .                               # collect all the updates done in the current directory
git commit -m "describe your updates"   # make a commit with your updates and a description
git push origin "my_branch"             # push your commit to your own, working branch
```

