#!/usr/bin/python3
def lexan():
	global mitr
	try:
		return(next(mitr))
	except StopIteration:
		return('')

def match(ch):
    global lookahead
    print(lookahead)
    if ch == lookahead:
        lookahead = lexan()
    else:
        print("ERROR")
        exit()

def oprnd():
	global lookahead

def cond():
	global lookahead

def factor():
	global lookahead

def term():
	global lookahead

def expr():
    global lookahead
    e = 0
    try:
        e = int(lookahead)
        match(lookahead)
    except:
        try:
            e = float(lookahead)
            match(lookahead)
        except:
            error()
    while(lookahead == '+' or lookahead == '-'):
        if lookahead == '+':
            match('+')
            e += expr()
        elif lookahead == '-':
            match('-')
            e -= expr()
    return e   

def type():
    global lookahead
    t = lookahead
    match(lookahead)
    if lookahead == '=':
        match('=')
        e = expr()
        if (t == 'int' and isinstance(int(e) , int)):
            return int(e)
        elif(t == 'real'  and isinstance(float(e) , float)):
            return float(e)
    error()

def decl():
    global lookahead
    global id
    curId = lookahead
    match(lookahead)
    if lookahead == ':':
        match(':')
        id[curId] = type()
        print(curId, ' = ', id[curId])
        match(lookahead)
    else:
        error()

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
        error()

def prog():
    global lookahead
    if lookahead == 'let':
        letInEnd()
        while lookahead == 'let':
            letInEnd()
    else:
        error()

def error():
    print("ERROR")
    exit()

import sys
print(sys.argv)
infile = open(sys.argv[1], 'r')
wlist = infile.read().split()
print(wlist)
mitr = iter(wlist)
lookahead = lexan()
id = {}
prog()
if lookahead != '':
    print("ERROR")