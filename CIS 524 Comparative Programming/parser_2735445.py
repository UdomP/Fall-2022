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

def type():
    global lookahead

def decl():
    global lookahead

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

def error(message):
    print(message)
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