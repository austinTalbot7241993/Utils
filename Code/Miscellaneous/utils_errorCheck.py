def check_float(number,domain,message):
    if (number>domain[1])|(number < domain[0]):
        myMessage = message + ' must be between ' + str(domain[0]) + \
                    ' and ' +  str(domain[1])
        raise ValueError(myMessage)

def check_string(myStr,myList,message):
    if myStr not in myList:
        myMessage = message + ' ' + myStr + ' invalid'
        raise ValueError(myMessage)

