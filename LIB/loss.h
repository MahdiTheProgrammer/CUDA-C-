//the structure of the model will be saved here
#include "utils.h"


class Loss{
public:
	float BCE(Tensor outputs, Tensor labels);
	float MSE(Tensor outputs, Tensor labels);
	float HuberLoss(Tensor outputs, Tensor labels);
}
