#pragma once

#include <iostream>

enum tile_t { BLANK, PLAYER, AI };
enum turn_t { T_PLAYER, T_AI };

extern tile_t board[9];
extern turn_t turn;
inline tile_t & index_board(int x, int y) noexcept { return board[y * 3 + x]; }

inline void game_init() {
    for (auto & i : board) i = BLANK;
    turn = T_PLAYER;
}

inline bool game_complete() {
    for (int y = 0; y < 3; y++) {
        if (
            index_board(0, y) == index_board(1, y) &&
            index_board(1, y) == index_board(2, y) &&
            index_board(2, y) == index_board(0, y)
        ) {
            if (index_board(0, y) != BLANK) return true;
        }
    }
    for (int x = 0; x < 3; x++) {
        if (
            index_board(x, 0) == index_board(x, 1) &&
            index_board(x, 1) == index_board(x, 2) &&
            index_board(x, 2) == index_board(x, 0)
        ) {
            if (index_board(x, 0) != BLANK) return true;
        }
    }
    return false;
}
inline turn_t game_winner() {
    for (int y = 0; y < 3; y++) {
        if (
            index_board(0, y) == index_board(1, y) &&
            index_board(1, y) == index_board(2, y) &&
            index_board(2, y) == index_board(0, y)
        ) {
            if (index_board(0, y) == PLAYER) return T_PLAYER;
            else if (index_board(0, y) == AI) return T_AI;
        }
    }
    for (int x = 0; x < 3; x++) {
        if (
            index_board(x, 0) == index_board(x, 1) &&
            index_board(x, 1) == index_board(x, 2) &&
            index_board(x, 2) == index_board(x, 0)
        ) {
            if (index_board(x, 0) == PLAYER) return T_PLAYER;
            else if (index_board(x, 0) == AI) return T_AI;
        }
    }
    return T_PLAYER;
}

inline void game_print() {
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++) {
            switch (index_board(x, y)) {
                case PLAYER: std::cout << "P "; break;
                case AI: std::cout << "A "; break;
                case BLANK: std::cout << "  "; break;
            }
        }
        std::cout << "\n";
    }
}