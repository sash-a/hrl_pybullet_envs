rm -rf build dist hrl_pybullet_env.egg-info

python3 -m build
python3 -m twine upload --repository pypi dist/*