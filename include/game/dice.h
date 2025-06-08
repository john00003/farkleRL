#pragma once

#include <array>

class die{
private:
	unsigned int value;
	bool isSetAside;
public:
	die();

	unsigned int roll();

	void setAside();
	
	void reset();
};

class dice{
private:
	std::array<die, 6> gameDice;
public:
	dice();
};

struct setAsideChoice{
	// choose to set aside die1
	bool die1;
	bool die2;
	bool die3;
	bool die4;
	bool die5;
	bool die6;
};
