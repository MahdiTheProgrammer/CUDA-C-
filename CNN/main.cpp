//Starts the program (creates and trains the model)
#include <iostream>
#include "utils.h"

int  main(){
	std::vector<int> shape = {2,3,4,5,5,3};
	Tensor t_A(shape);
	t_A.arrange(1,1);
	t_A.print();
	return 0;
}
