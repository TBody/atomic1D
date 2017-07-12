import json

data = {}
data['a'] = 1
data['b'] = [2,3,4]
data['c'] = 'abc'

print(data)

with open('data.json','w') as fp:
	json.dump(data, fp, sort_keys=True, indent=4)

data = None

with open('data.json','r') as fp:
	data = json.load(fp)

print(data)