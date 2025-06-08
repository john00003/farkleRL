#pragma once

#include <random>
#include <array>
#include <vector>

class turnState;
#include "player.h"
#include "dice.h"

class gameState{
    // a model class for the game
private:
	unsigned int numPlayers;
	unsigned int turn;
	std::vector<player> players;
	dice gameDice;

public:
	gameState(unsigned int numPlayers);
	gameState(std::vector<player>& players);
	int checkIfWin() const;		// check a player has won, returning player number if so, otherwise returning -1
	void endTurn();
};

class turnState{
private:
	gameState& state;
	unsigned int pointsThisTurn;

public:
	turnState(gameState& state);
};

class game{
	// a controller class for the game
private:
	gameState state;

	
public:
	game(unsigned int numPlayers);		// for initializing with default implementation of players
	game(std::vector<player>& players);	// for initializing with a select implementation of players
	bool checkIfLegal(dice dice, setAsideChoice choice);
	int checkPointValue(dice dice, setAsideChoice choice);


};

