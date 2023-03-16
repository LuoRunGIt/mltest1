
a=10
b=10
print(id(a),id(b))

c=[1,2,3,4,5]
#d=c.extend()
#print(id(c),id(d))

sum=0
for ele in range (1,100):
    if ele%2==0:
        sum+=ele
        print(sum)

l = ['abd','ewetrwef','fered','qwszfgfsaffs']
l.sort(key=lambda ele:ele[len(ele)-1],reverse=True)
print(l)