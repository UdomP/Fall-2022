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

def cond():
	global lookahead

def factor():
	global lookahead

def term():
	global lookahead

def expr():
	global lookahead

def type():
	global lookahead

def decl():
    global lookahead
    global d
    id = lookahead
    match(lookahead)
    if lookahead == ':':
        match(':')
        t = type()
        if lookahead == '=':
            match('=')
            exp = lookahead
            if exp.isdigit():
                if (isinstance(int(exp) , int)) and (t == 'int'):
                    d[id] = expr(t)
            elif '.' in exp:
                if (isinstance(float(exp), float) and t == 'real'):
                    d[id] = expr(t)
            elif exp.isalpha():
                d[id] = expr(t)
            if lookahead == ';':
                match(';')
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
            type()
            if lookahead == '(':
                match('(')
                expr()
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
mitr = iter(wlist)
lookahead = lexan()
d = {}
prog()
if lookahead != '':
    print("ERROR")