/*=========================================================================

     Copyright (c) 2018-2019 IRSTEA
     Copyright (c) 2020-2021 INRAE


     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include "otbTestMain.h"

void RegisterTests()
{
  REGISTER_TEST(floatValueToTensorTest);
  REGISTER_TEST(intValueToTensorTest);
  REGISTER_TEST(boolValueToTensorTest);
  REGISTER_TEST(floatVecValueToTensorTest);
  REGISTER_TEST(intVecValueToTensorTest);
  REGISTER_TEST(boolVecValueToTensorTest);
}

