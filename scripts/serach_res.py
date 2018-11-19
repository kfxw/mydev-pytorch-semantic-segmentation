import glob,re

p=re.compile('Mean IoU: 0.52')
a=glob.glob('*trainset*ade*.log')

for i in a:
	b=p.findall(file(i).read())
	if len(b)>0:
		print i
		print b
