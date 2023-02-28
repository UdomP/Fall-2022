#!/usr/bin/python3
def lexan():
	global mitr
	try:
		return(next(mitr))
	except StopIteration:
		return('')

def match(ch):
    global lookahead
    if ch == lookahead:
        lookahead = lexan()
    else:
        print("ERROR")
        exit()

def oprnd():
    global lookahead
    if lookahead in id:
        curId = lookahead
        match(curId)
        return id[curId]
    elif isInt(lookahead) or isFloat(lookahead):
        try:
            temp = int(lookahead)
            match(lookahead)
            return temp
        except:
            pass
    error('Error oprnd()')

def cond():
    global lookahead
    curOp = oprnd()
    if lookahead == '<':
        match('<')
        return curOp < oprnd()
    elif lookahead == '<=':
        match('<=')
        return curOp <= oprnd()
    elif lookahead == '>':
        match('>')
        return curOp > oprnd()
    elif lookahead == '>=':
        match('>=')
        return curOp >= oprnd()
    elif lookahead == '==':
        match('==')
        return curOp == oprnd()
    elif lookahead == '<>':
        match('<>')
        return curOp != oprnd()
    error('Error cond()')

def factor():
    global lookahead
    if lookahead == '(':
        match('(')
        temp = expr()
        if lookahead == ')':
            match(')')
            return temp
    elif lookahead in id:
        temp = lookahead
        match(lookahead)
        return id[temp]
    elif lookahead == 'real' or lookahead == 'int':
        curType = lookahead
        match(curType)
        if lookahead == '(':
            match('(')
            curId = lookahead
            match(curId)
            temp = type(curType, id[curId])
            if lookahead == ')':
                match(')')
                return temp
    elif isInt(lookahead) or isFloat(lookahead):
        try:
            temp = int(lookahead)
            match(lookahead)
            return temp
        except:
            try:
                temp = float(lookahead)
                match(lookahead)
                return temp
            except:
                pass
    error('Error factor()')
        

def term():
    global lookahead
    val = factor()
    while lookahead == '*' or lookahead == '/':
        if lookahead == '*':
            match('*')
            val *= term()
        if lookahead == '/':
            match('/')
            val /= term()
    return val

def expr():
    global lookahead
    if lookahead == 'if':
        match('if')
        curCond = cond()
        if lookahead == 'then':
            match('then')
            curThen = expr()
            if lookahead == 'else':
                match('else')
                curElse = expr()
            if curCond:
                return curThen
            else:
                return curElse
    else:
        val = term()
        while lookahead == '+' or lookahead == '-':
            if lookahead == '+':
                match('+')
                val += term()
            if lookahead == '-':
                match('-')
                val -= term()
        return val
    error('Error expr()')

def type(t, e):
    global lookahead
    if t == 'real':
        return float(e)
    elif t == 'int':
        return int(e)

def decl():
    global lookahead
    curId = lookahead
    match(curId)
    if lookahead == ':':
        match(':')
        curType = lookahead
        match(lookahead)
        if lookahead == '=':
            match('=')
            id[curId] = type(curType, expr())
            if lookahead == ';':
                match(';')
                return
    error('Error decl()')

def declList():
    global lookahead
    decl()
    while lookahead != 'in':
        decl()

def letInEnd():
    global lookahead
    if lookahead == 'let':
        match('let')
        declList()
        if lookahead == 'in':
            match('in')
            curType = lookahead
            match(lookahead)
            if lookahead == '(':
                match('(')
                print(type(curType, expr()))
                if lookahead == ')':
                    match(')')
                    if lookahead == 'end':
                        match('end')
                        if lookahead == ";":
                            match(';')
    else:
        error('Error letInEnd()')


def prog():
    global lookahead
    if lookahead == 'let':
        letInEnd()
        while lookahead == 'let':
            letInEnd()
    else:
        error('Error prog()')

def isInt(s):
    try:
        int(s)
        return True
    except:
        return False

def isFloat(s):
    try:
        float(s)
        return True
    except:
        return False

def error(message):
    print(message)
    exit()

import sys
print(sys.argv)
infile = open(sys.argv[1], 'r')
# infile = open('CIS 524 Comparative Programming/sample.tiny', 'r')
wlist = infile.read().split()
mitr = iter(wlist)
lookahead = lexan()
id = {}
prog()
if lookahead != '':
    print("ERROR")