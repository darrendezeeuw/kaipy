In order to push the private master branch over to the public master branch follow these steps.  Note the removal of this file is to prevent it from being included in the public repo.

```bash
mkdir kaipy-promotion
cd kaipy-promotion
git clone git@bitbucket.org:aplkaiju/kaipy-private.git
cd kaipy-private
rm Private2PublicSteps.md
git remote add public git@bitbucket.org:aplkaiju/kaipy.git
git fetch
git push --force public master
```