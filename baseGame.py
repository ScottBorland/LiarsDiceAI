import random

#variables for each player's set of dice

p1dice = []
p2dice = []

p1numDice = 5
p2numDice = 5

allDice = []

#variable to keep track of who's turn it is

playerTurn = 1

#variable to keep track of how many turns there have been

turn = 0

#'bid' is the last bet made by a player. A claim of 4,3 means the last player has claimed there are at least 4 threes

bid = []
prevBid = []

#Check if bid is valid and return true if so, false if not

def validateBid(bid, allDice):
    freq = bid[1]
    num = bid[0]
    print("All dice:")
    print(allDice)

    numberOfOnes = allDice.count(1)

    if(num == 1):
        totalOfNum = numberOfOnes
    else:
        totalOfNum = allDice.count(1) + allDice.count(num)
        print("Total number:")
        print (totalOfNum)

    if(freq > totalOfNum):
        print("frequency:")
        print (freq)
        print("Total of num:")
        print (totalOfNum)
        return False
    else:
        return True

#Randomise dice to start game
def rollDice():
    p1dice.clear()
    p2dice.clear()

    global allDice
    
    x = range(p1numDice)
    for n in x:
        p1dice.append(random.randint(1, 6))
    y = range(p2numDice)
    for x in y:
        p2dice.append(random.randint(1, 6))
    allDice = p1dice + p2dice

def checkIfBidAllowed(bid):
    number = bid[0]
    frequency = bid[1]

    prevNum = prevBid[0]
    prevFrequency = prevBid[1]

    if number >= prevNum:
        if number == prevNum:
            if frequency <= prevFrequency:
                return False
            else:
                return True
        else:
            return True
    else:
        return False

def makeBid():
    numBid = input("What number are you bidding on?")
    freqBid = input("How many are you bidding?")

    bid = (int(numBid), int(freqBid))

    global playerTurn
    global turn
    global prevBid

    if(checkIfBidAllowed(bid) == False):
        print("Invalid bid. Please bid again")
        makeBid()
    else:
        prevBid = bid
        if(playerTurn == 1):
            playerTurn = 2
        else:
            playerTurn = 1
        statePlayerTurn()
        turn += 1
        chooseIfChallenge()
        

def showDice():
    print("Player 1 Dice: ")
    print(p1dice)
    print("Player 2 Dice: ")
    print(p2dice)

def chooseIfChallenge():
    print('Do you want to challenge?')
    response = input('y for yes, n for no')
    if(response == 'y'):
        challenge()
    elif (response == 'n'):
        makeBid()
    else:
        print('Invalid response. Please respond with y or n')
        chooseIfChallenge()

def statePlayerTurn():
    global playerTurn
    print("Player " + str(playerTurn) + "'s turn")

def game():
    
    rollDice()
    showDice()    

    global allDice
    allDice = p1dice + p2dice

    global playerTurn
    playerTurn = 1
    
    global prevBid
    prevBid = [1,0]

    print("Player " + str(playerTurn) + "'s turn")

    global turn
    
    if(turn != 0):
        chooseIfChallenge()
    else:
        makeBid()
    
    turn += 1
    
def challenge():
    global prevBid
    global playeTurn
    if(validateBid(prevBid, allDice)):
        if(playerTurn == 1):
            print ("Bid is true. Player 2 wins")
        else:
            print ("Bid is true. Player 1 wins")
    else:
        if(playerTurn == 1):
            print("Bid is false. Player 2 wins")
        else:
            print("Bid is false. Player 1 wins")
        game()
    
game()

    


    
    

