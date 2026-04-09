#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <tuple>
#include <utility>
#include <cstdlib>
#include <chrono>

void clear_screen() {
#ifdef _WIN32
    std::system("cls");
#else
    // Assume Unix-like system
    std::system("clear");
#endif
}

// initializaton of preset register
struct option_parameters {
    std::string option_family;
    std::string option_type;
    double S_0;
    double K;
    double r;
    double T;
    double sigma;
    int n_steps;
    int n_branches;
    double q;
};

std::string get_input(std::string message){
    std::cout << message;
    std::string user_input{};
    std::cin >> user_input;
    return user_input;
}
double option(std::string option_family, std::string option_type, double S_0, double K, double r, double T, double sigma, int n_steps, int n_branches, double q = 0) {
    if (n_branches != 2 && n_branches != 3) {
        throw std::invalid_argument("Valid choice of binomial (n=2) or trinomial (n=3) tree is required.");
    }
    if (S_0 <= 0 || K <= 0 || r <= 0 || T <= 0 || sigma <= 0 || n_steps < 1) {
        throw std::invalid_argument("Pricing parameters must be non-zero; steps should be greater or equal to one.");
    }
    if (n_branches == 2) {
        double dt {T / n_steps};
        double up {exp(sigma * sqrt(dt))};
        double down {1 / up};
        double p {(exp((r-q) * dt) - down) / (up - down)};
       
        std::function<double(double)> payoff_func; // relieves local scope issues with lambda function declaration

        if (option_type == "call") {
            payoff_func = [K](double S_n) { return std::max(S_n - K, 0.0); };
        } else {
            payoff_func = [K](double S_n) { return std::max(K - S_n, 0.0); };
        }        

        // final time step loop
        std::vector<double> node_values(n_steps+1);
        for(int i=0; i<n_steps+1; ++i) {
            double log_S_n {std::log(S_0) + (n_steps - i)*std::log(up) + i * std::log(down)};
            double S_n {exp(log_S_n)};
            node_values[i] = std::max(payoff_func(S_n), static_cast<double>(0));
        }
        // rest...
        if (option_family=="European") {
            for(int i=n_steps-1; i >= 0; --i) {
                for (int j=0; j <= i; ++j) {
                    node_values[j] = exp(-r * dt) * (p * node_values[j] + (1 - p) * node_values[j+1]);
                }  
            }
        }
        else if (option_family=="American") {
            for(int i=n_steps-1; i >= 0; --i) { // variable initialization within loop incurs no overhead due to compiler optimizations
                for (int j=0; j <= i; ++j) {
                    double european_value = exp(-r * dt) * (p * node_values[j] + (1 - p) * node_values[j + 1]);
                    double log_S_j {std::log(S_0) + (double)(i - 2 * j) * std::log(up)};
                    double S_j = exp(log_S_j);
                    double intrinsic_value = payoff_func(S_j);
                    node_values[j] = std::max(european_value, intrinsic_value);
                }
            }
        }
    return node_values[0];
    }
    else {
        return 0.0;
    }
}
bool historical_menu(option_parameters reg_params, std::chrono::duration<double> calibration) {
    // should add adjustment ability
    while (true) {
        std::cout   << "-----------------------------" << '\n';
        std::cout   << "(" << reg_params.option_family << " " << reg_params.option_type << ")" << '\n' << "Trade Price: " << reg_params.S_0 << "GP" << '\n' << "Strike: " 
                    << reg_params.K << "GP" << '\n' << "Risk-Free: " << reg_params.r*100 << '%' << '\n' << "Dividend Yield: " << reg_params.q*100 << '%' << '\n' 
                    << "Time to Expiry (years): " << reg_params.T << '\n' << "Historical Volatility: " << reg_params.sigma << '\n'
                    << "Tree Layers: " << reg_params.n_steps << '\n' << "Node Branches: " << reg_params.n_branches << '\n'; 
        std::cout   << "-----------------------------" << '\n';
        double calibration_seconds {calibration.count()};
        std::cout   << "Benchmarked Estimate Runtime: "<< calibration_seconds * pow(static_cast<double>(reg_params.n_steps)/1000,2) << '\n' << "Saved parameters. Confirm? (y/n): "; // 1000 step calibrated benchmark

        std::string user_input{};
        std::cin >> user_input;
        if (user_input == "y") {
            double option_value = option(
                        reg_params.option_family,
                        reg_params.option_type,
                        reg_params.S_0,
                        reg_params.K,
                        reg_params.r,
                        reg_params.T,
                        reg_params.sigma,
                        reg_params.n_steps,
                        reg_params.n_branches,
                        reg_params.q
                    );
                    clear_screen();
                    std::cout << "Calculated Option Price: " << option_value << " GP" <<'\n' << "----------------------------" << '\n';
                    return true;
        }
        else if (user_input == "n") {
            clear_screen();
            return false;
        } else {
            clear_screen();
            continue;
        }
    }
}
bool option_menu(std::string option_family, std::string option_type, option_parameters& reg_params, std::chrono::duration<double> calibration){
    std::cout << "<<< <b>" << '\n';
    while (true) {
        std::string S_0_raw         {get_input("Current Asset Price: ")};
        if (S_0_raw=="b")           return false;
        std::string K_raw           {get_input("Strike Price: ")};
        if (K_raw=="b")             return false;
        std::string r_raw           {get_input("Risk Free Rate: ")};
        if (r_raw=="b")             return false;
        std::string T_raw           {get_input("Time to expiry (Years): ")};
        if (T_raw=="b")             return false;
        std::string sigma_raw       {get_input("Historical Volatility: ")};
        if (sigma_raw=="b")         return false;
        std::string n_steps_raw     {get_input("Tree Layers: ")};
        if (n_steps_raw=="b")       return false;
        std::string n_branches_raw  {get_input("Node Branches (binomial is only supported currently): ")};
        if (n_branches_raw=="b")    return false;
        clear_screen();
        
        try {
            reg_params.option_family = option_family;
            reg_params.option_type =   option_type;
            reg_params.S_0 =           std::stod(S_0_raw);
            reg_params.K =             std::stod(K_raw);
            reg_params.r =             std::stod(r_raw);
            reg_params.T =             std::stod(T_raw);
            reg_params.sigma =         std::stod(sigma_raw);
            reg_params.n_steps =       std::stoi(n_steps_raw);
            reg_params.n_branches =    std::stoi(n_branches_raw);
            reg_params.q =              0.0; // dividend yield; defaults to 0
            
            historical_menu(reg_params, calibration);
            // double option_value{option(option_family, option_type, reg_params.S_0, reg_params.K, reg_params.r, reg_params.T, reg_params.sigma, reg_params.n_steps, reg_params.n_branches, reg_params.q)};
            // std::cout << "Calculated Option Price: " << option_value << " GP" <<'\n';
            break;
        }
        catch (const std::exception& e) { // can catch std::stod out_of_range | american_option() invalid_argument
            std::cout <<"Exception: " << e.what();   
            continue;
        }
    }
    return true;
}
bool main_menu_2(std::string option_type, option_parameters& reg_params, std::chrono::duration<double> calibration) {
    while (true) {
        std::cout << "Select Option Type:" << '\n';
        std::cout << "<1> American" << '\n' << "<2> European" << '\n' << "<3> Back" << '\n';
        std::string option_selection_str{};
        std::cin >> option_selection_str;
        int option_selection{};
        try {
            option_selection = std::stoi(option_selection_str);
        }
        catch(...){
            clear_screen();
            continue;
        }
        switch (option_selection) {
            case 1: 
                clear_screen();
                if (option_menu("American", option_type, reg_params, calibration)) {
                return true;
                }
                break;
                    
            case 2: 
                clear_screen();
                if (option_menu("European", option_type, reg_params, calibration)) {
                    return true;
                }
            case 3:
                clear_screen();
                return false;
            default:
                clear_screen();
                continue;
        }
    }
}
bool main_menu(std::chrono::duration<double> calibration) {
    option_parameters reg_params;
    // sentinel value to indicate non-initialization
    reg_params.n_steps = 0; 

    std::cout << "-----------------------------" << '\n' << "OSRS Options Calculation v0.1" << '\n' << "-----------------------------" << '\n';
    while (true) {
        std::cout << "Select Option Type:" << '\n';
        std::cout << "<1> Call Option" << '\n' << "<2> Put Option" << '\n' << "<3> Previous Option" << '\n' << "<4> Quit" << '\n';
        std::string option_selection_str{};
        std::cin >> option_selection_str;
        int option_selection{};
        try {
            option_selection = std::stoi(option_selection_str);
        }
        catch(...){
            continue;
        }
        switch (option_selection) {
            case 1:
                clear_screen();
                main_menu_2("call", reg_params, calibration);
                continue;
            case 2:
                clear_screen();
                main_menu_2("put", reg_params, calibration);
                continue;
            case 3:
                {
                    // check if the struct has been initialized
                    if (reg_params.n_steps == 0) {
                        clear_screen();
                        std::cout << "No previous option data available. Please run an option calculation first." << '\n' 
                        << "--------------------------------------------------------------------------" << '\n';
                        continue;
                    }
                    clear_screen();
                    if (historical_menu(reg_params, calibration)) {
                        continue;
                    }
                }
                break;
            case 4:
                clear_screen();
                std::cout << "Exiting..." << '\n';
                return false;
            default:
                clear_screen();
                continue;
        }
    }
    return true;
}

int main(){
    clear_screen();
    std::cout << "Running Calibration..";
    const int n_cal = 1000;
    auto start_time = std::chrono::high_resolution_clock::now();
    option("American", "call", 1000, 1500, 0.02, 1, 1, n_cal, 2);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> calibration_duration = end_time - start_time; //double type defaults to seconds
    clear_screen();

    bool keep_running{main_menu(calibration_duration)};
    if (!keep_running) {
    return 0;
    }
    return 1;
}