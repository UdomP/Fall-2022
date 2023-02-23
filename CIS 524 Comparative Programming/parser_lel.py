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
	global d
	if lookahead.isalpha():
		temp = lookahead
		match(lookahead)
		return d[temp]
	elif lookahead.isdigit() or "." in lookahead:
		temp = lookahead
		match(lookahead)
		if temp.isdigit():
			if (isinstance(int(temp) , int)):
				return int(temp)
		elif '.' in temp:
			if (isinstance(float(temp), float)):
				return float(temp)
		else:
			print("ERROR")
def cond():
	global lookahead
	global d
	op1 = oprnd()
	temp = lookahead
	match(lookahead)
	op2 = oprnd()
	if temp == "<":
		return op1 < op2
	if temp == "<=":
		return op1 <= op2
	if temp == ">":
		return op1 > op2
	if temp == ">=":
		return op1 >= op2
	if temp == "==":
		return op1 == op2
	if temp == "<>":
		return op1 != op2
	else:
		print("ERROR")
		exit()

def factor(exp_typ):
	global lookahead
	global d
	if lookahead == '(':
		match('(')
		temp = expr(exp_typ)
		if lookahead == ')':
			match(')')
		return temp
	elif lookahead.isalpha() and lookahead != 'int' and lookahead != 'real':
		temp = lookahead
		match(lookahead)
		return d[temp]
	elif lookahead.isdigit() or "." in lookahead:
		temp = lookahead
		match(lookahead)
		if temp.isdigit():
			if (isinstance(int(temp) , int)) and (exp_typ == 'int'):
				return int(temp)
		elif '.' in temp:
			if (isinstance(float(temp), float) and exp_typ == 'real'):
				return float(temp)
	elif lookahead == 'int' or lookahead == 'real':
		t = type()
		if lookahead == '(':
			match('(')
			if lookahead.isalpha():
				temp = lookahead
				match(lookahead)
				if t == 'int':
					return int(d[temp])
				elif t == 'real':
					return float(d[temp])
			if lookahead == ')':
				match(')')
		return d[temp]
	else:
		print("ERROR")
		exit()

def term(exp_typ):
	global lookahead
	global d
	id1 = factor(exp_typ)
	while lookahead == '*' or lookahead == '/':
		if lookahead == '*':
			match('*')
			id2 = factor(exp_typ) 
			id1 = id1 * id2 
		elif lookahead == '/':
			match('/')
			id2 = factor(exp_typ)
			id1 = id1 / id2 
		else:
			print("ERROR")
			exit()
	return id1

def expr(exp_typ):
	global lookahead
	global d
	if lookahead != 'if':
		id1 = term(exp_typ)
		while lookahead == '+' or lookahead == '-':
			if lookahead == '+':
				match('+')
				id2 = term(exp_typ)
				id1 = id1 + id2 
			elif lookahead == '-':
				match('-')
				id2 = term(exp_typ)
				id1 = id1 - id2 
			else:
				print("ERROR")
				exit()
		return id1
	elif lookahead == 'if':
		match('if')
		boo = cond()
		if lookahead == 'then':
			match('then')
		op1 = expr(exp_typ)
		if lookahead == 'else':
			match('else')
		op2 = expr(exp_typ)
		if boo:
			return op1
		elif boo != True:
			return op2
	else:
		print("ERROR")
		exit()

def type():
	global lookahead
	if lookahead == 'int':
		match(lookahead)
		return 'int'
	elif lookahead == 'real':
		match(lookahead)
		return 'real'
	else:
		print("ERROR")
		exit()

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
		print("ERROR")
		exit()

def declList():
	global lookahead
	decl()
	while lookahead != 'in':
		decl()

def LetInEnd():
	global lookahead
	if lookahead == 'let':
		match('let')
		declList()
		if lookahead == 'in':
			match('in')
		t = type()
		if lookahead == '(':
			match('(')
		temp = (expr(t))
		while lookahead == '+' or lookahead == '-' or lookahead == '*' or lookahead == '/' or lookahead == ')':
			if lookahead == '+':
				match('+')
				temp = temp + expr(t)
			if lookahead == '-':
				match('-')
				temp = temp - expr(t)
			if lookahead == '*':
				match('*')
				temp = temp * expr(t)
			if lookahead == '/':
				match('/')
				temp = temp / expr(t)
			if lookahead == ')':
				match(')')
		if lookahead == 'end':
			match('end')
		if lookahead == ';':
			match(';')
		if t == 'int':
			print(int(temp))
		elif t == 'real':
			print(float(temp))
	else:
		print("ERROR")

def prog():
	global lookahead
	if lookahead == 'let':
		LetInEnd()
		while lookahead == 'let':
			LetInEnd()
	else:
		print("ERROR")
		exit()

import sys
infile = open(sys.argv[1], 'r')
# infile = open('CIS 524 Comparative Programming/sample.tiny', 'r')
wlist = infile.read().split()
mitr = iter(wlist)
lookahead = lexan()
d = {}
prog()
print(lookahead)
if lookahead != '':
    print("ERROR")