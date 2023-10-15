#include "tokenizer.hpp"

int main() {
    static PKLM::Model model;

    std::cout << "Loading network... ";
    model.load("pklm.net");
    std::cout << "Done!\n";

    model.give_data("HELLO", "HELLO");
    model.give_data("HI", "HELLO");
    model.give_data("HOW ARE YOU", "GOOD");
    model.give_data("HOW IS THE WEATHER", "IT IS GOOD");
    model.give_data("ABCD", "WXYZ");

//    for (int i = 0; i < 100; i++) {
//        model.train();
//        model.save("pklm.net");
//        std::cout << "Total cost: " << model.cost() << "\n";
//    }
//
//    model.save("pklm.net");

    while (true) {
        std::string input;
        std::getline(std::cin, input);

        std::cout << "Gave \"" << input << "\", got \"" << model.get_response(input.c_str()) << "\"\n";
    }
}