#pragma once

class player;
#include "game.h"
#include "dice.h"

class player{
private:
	const unsigned int playerNum; // rather than identifying by name, we use a number
	unsigned int score;


public:
	player(unsigned int playerNum);
	void addPoints();
	void checkPoints() const;
	setAsideChoice decideToSetAside(const turnState state) const;	
	bool decideToRoll(const turnState state) const;		// decide to either bank, or roll the remaining dice
	
};

