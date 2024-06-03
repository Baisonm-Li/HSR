
import os
import shutil

if __name__ == "__main__":
    discardFileList = ['__pycache__','out','.git','wandb','.gitignore','trained_modles']
    projectName = input("input your project name:")
    projectName = projectName.strip()
    projectDir = f'../{projectName}'
    os.mkdir(projectDir)
    fileAndDirList = [path for path in os.listdir('.') if path not in discardFileList]
    for p in fileAndDirList:
        shutil.copy(os.path.join('./',p), os.path.join(projectDir,p))
    print('finished')