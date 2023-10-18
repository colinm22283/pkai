#include <pkai/host.hpp>
#include <pkai/universal/network_builder.hpp>
#include <pkai/universal/connection/fully_connected.hpp>
#include <pkai/universal/activation_function/relu.hpp>

#include "game.hpp"

using namespace PKAI;
using namespace Connection;
using namespace ActivationFunction;

using Builder = NetworkBuilder
    ::DefineFloatType<float>
    ::AddLayer<9 * 3>
    ::AddConnection<FullyConnected<ReLu>>
    ::AddLayer<50>
    ::AddConnection<FullyConnected<ReLu>>
    ::AddLayer<9>;

static Builder::Network network;
static Builder::Dataset dataset;

inline void play_game() {
    game_init();

    game_print();

    while (!game_complete()) {
        switch (turn) {
            case T_PLAYER: {
                std::cout << "Player Turn:\n";
                int x, y;
                std::cout << "X: ";
                std::cin >> x;
                std::cout << "Y: ";
                std::cin >> y;

                index_board(x, y) = PLAYER;

                turn = T_AI;
            } break;
            case T_AI: {
                std::cout << "AI Turn:\n";
                turn = T_PLAYER;
            } break;
        }

        game_print();
    }

    switch (game_winner()) {
        case T_PLAYER: std::cout << "Player win\n"; break;
        case T_AI: std::cout << "AI win\n"; break;
    }
}

int main() {
    while (true) {
        play_game();
    }
}