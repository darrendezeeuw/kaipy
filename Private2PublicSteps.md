In order to push the private master branch over to the public master branch follow these steps.

```bash
mkdir kaipy-promotion
cd kaipy-promotion
git clone git@bitbucket.org:aplkaiju/kaipy-private.git
cd kaipy-private
git remote add public git@bitbucket.org:aplkaiju/kaipy.git
git fetch
git push --force public master
```