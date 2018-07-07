#Python testing

side = 5
print('Welcome to ML Word')
for x in list(range(side)) + list(reversed(range(side-1))):
    print('{: <{w1}}{:*<{w2}}'.format('', '', w1=side-x-1, w2=x*2+1))

print('Asela Wijesinghe - AS2015606')

#/////////////////////////////// real codes
