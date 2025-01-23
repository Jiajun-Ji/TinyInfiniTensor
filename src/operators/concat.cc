#include "operators/concat.h"
#include "utils/operator_utils.h"
#include <cstddef>

namespace infini {
ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim) : OperatorObj(OpType::Concat, inputs, {output}) {
    int rank = inputs[0]->getRank();
    dim = get_real_axis(_dim, rank);
    IT_ASSERT(checkValid(graph));
}

optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs) {
    // printf("concat func begin\r\n");
    Shape dims = inputs[0]->getDims();
    auto rank = inputs[0]->getRank();
    // inputs.max_size()

    // =================================== 作业 ===================================
    // TODO：修改 dims，返回正确的 concat 后的 shape
    // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
    // =================================== 作业 ===================================
    // printf("concat loop begin\r\n");
    Shape dims_output = inputs[0]->getDims();

    // printf("concat loop begin\r\n");
    for (size_t i = 1; i < inputs.size(); ++i) {
    	// printf("concat loop %ld\r\n",i);
    	dims_output[dim] += inputs[i]->getDims()[dim];
    }

	return vector<Shape>{dims_output};

    // Shape outputShape = inputs[0]->getDims();
    // size_t concatDim = static_cast<size_t>(getDim()); // 确保 concatDim 是 size_t 类型

    // // 遍历输入张量以检查维度
    // for (size_t i = 1; i < inputs.size(); i++) {
    //     const Shape& currentShape = inputs[i]->getDims();
        
    //     // 验证所有张量在除拼接维度外的形状相同
    //     for (size_t d = 0; d < outputShape.size(); d++) {
    //         if (d != concatDim && outputShape[d] != currentShape[d]) {
    //             IT_TODO_HALT(); // 如果形状不匹配，抛出异常
    //         }
    //     }

    //     // 累加拼接维度的大小
    //     outputShape[concatDim] += currentShape[concatDim];
    // }

    // return vector<Shape>{outputShape}; // Return the calculated output shape

    // return vector<Shape>{dims_output};

    // Shape dims_input2 = inputs[1]->getDims();
    // dims_input2[dim] = dims_input2[dim] + dims[dim];

    // return vector<Shape>{dims_input2};
    // return {{dims}};
}

std::string ConcatObj::toString() const {
    std::ostringstream os;
    os << "Concat[" << getGuid() << "]";
    os << "(";
    for (auto input : inputs)
        os << vecToString(input->getDims()) << ",";
    os << "dim=" << dim << ",";
    os << "input=";
    for (auto input : inputs)
        os << input->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

} // namespace infini
