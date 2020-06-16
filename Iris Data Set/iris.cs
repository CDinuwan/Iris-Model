using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Iris_Data_Set
{
    class iris
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(0)]
        public float SepalWidth;

        [LoadColumn(0)]
        public float PetalLength;

        [LoadColumn(0)]
        public float PetalWidth;
    }
}
