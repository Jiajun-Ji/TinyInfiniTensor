#include "core/graph.h"
#include "core/op_type.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <queue>

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator &op) {
    sorted = false;
    ops.push_back(op);
    for (auto &input : op->getInputs()) {
        if (input) {
            input->addTarget(op);
            if (auto pred = input->getSource()) {
                pred->addSuccessors(op);
                op->addPredecessors(pred);
            }
        }
    }
    for (auto &output : op->getOutputs()) {
        if (output) {
            output->setSource(op);
            for (auto &succ : output->getTargets()) {
                succ->addPredecessors(op);
                op->addSuccessors(succ);
            }
        }
    }
}

string GraphObj::toString() const {
    std::ostringstream oss;
    oss << "Graph Tensors:\n";
    for (const auto &tensor : tensors)
        oss << tensor << "\n";

    oss << "Graph operators:\n";
    for (const auto &op : ops) {
        vector<UidBaseType> preds, succs;
        for (auto &o : op->getPredecessors())
            preds.emplace_back(o->getGuid());
        for (auto &o : op->getSuccessors())
            succs.emplace_back(o->getGuid());
        oss << "OP " << op->getGuid();
        oss << ", pred " << vecToString(preds);
        oss << ", succ " << vecToString(succs);
        oss << ", " << op << "\n";
    }
    return oss.str();
}

bool GraphObj::topo_sort() {
    if (this->sorted) {
        return true;
    }
    std::vector<Operator> sorted;
    std::unordered_set<OperatorObj *> flags;
    sorted.reserve(ops.size());
    flags.reserve(ops.size());
    while (sorted.size() < ops.size()) {
        // Any node is move to sorted in this loop.
        auto modified = false;
        for (auto const &op : ops) {
            if (auto const &inputs = op->getInputs();
                flags.find(op.get()) == flags.end() && std::all_of(inputs.begin(), inputs.end(), [&flags](auto const &input) {
                    auto ptr = input->getSource().get();
                    return !ptr || flags.find(ptr) != flags.end();
                })) {
                modified = true;
                sorted.emplace_back(op);
                flags.insert(op.get());
            }
        }
        if (!modified) {
            return false;
        }
    }
    this->ops = std::move(sorted);
    return this->sorted = true;
}

void GraphObj::optimize() {
    // =================================== 作业 ===================================
    // TODO: 设计一个算法来实现指定的图优化规则
    // 图优化规则如下：
    // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
    // 2.
    // 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
    // =================================== 作业 ===================================
    for (auto it = ops.begin(); it != ops.end();) {
        auto currentOp = *it;
        if (currentOp->getOpType() == OpType::Transpose) {
            // 获取当前 Transpose 算子
            auto currentTransposeOp = as<TransposeObj>(currentOp);

            // 获取它的前一个算子（如果存在）并判断其类型
            auto inputTensor = currentTransposeOp->getInputs()[0];
            auto previousOp = inputTensor ? inputTensor->getSource() : nullptr;

            if (previousOp && previousOp->getOpType() == OpType::Transpose) {
                auto previousTransposeOp = as<TransposeObj>(previousOp);

                // 判断两个 Transpose 是否是逆操作
                auto prevPermute = previousTransposeOp->getPermute();
                auto currPermute = currentTransposeOp->getPermute();

                bool isInverse = true;
                if (prevPermute.size() == currPermute.size()) {
                    for (size_t i = 0; i < prevPermute.size(); ++i) {
                        if (prevPermute[currPermute[i]] != static_cast<int>(i)) {
                            isInverse = false;
                            break;
                        }
                    }
                } else {
                    isInverse = false;
                }

                if (isInverse) {
                    // 删除冗余的两个 Transpose 算子
                    auto outputTensor = currentTransposeOp->getOutputs()[0];
                    auto inputTensor = previousTransposeOp->getInputs()[0];

                    // 更新连接关系
                    if (outputTensor && inputTensor) {
                        outputTensor->setSource(nullptr);               // 清除当前算子的来源
                        inputTensor->removeTarget(previousTransposeOp); // 从源张量移除目标

                        // 将原始输入的目标对接至当前的后继算子
                        for (auto &targetOp : outputTensor->getTargets()) {
                            targetOp->replaceInput(outputTensor, inputTensor);
                            inputTensor->addTarget(targetOp);
                        }
                    }

                    // 删除两个算子
                    removeOperator(previousTransposeOp);
                    removeOperator(currentTransposeOp);
                    // it = ops.erase(std::find(ops.begin(), ops.end(), previousTransposeOp));
                    // it = ops.erase(it);

                    if (outputTensor) {
                        removeTensor(outputTensor);
                    }
                    auto intermediateTensor = currentTransposeOp->getInputs()[0];
                    if (intermediateTensor && intermediateTensor != inputTensor) {
                        removeTensor(intermediateTensor);
                    }

                    continue; // 跳过递增，直接处理下一个算子
                }
            }
        }

        ++it; // 若无需删除，则处理下一个算子
    }

    // ==========================
    // 规则 2: 合并 Transpose 和 MatMul 算子
    // ==========================
    for (auto it = ops.begin(); it != ops.end();) {
        auto currentOp = *it;

        // 打印当前正在处理的操作符详细信息
        std::cout << "Processing Operation: " << currentOp->toString() << std::endl;

        if (currentOp->getOpType() == OpType::MatMul) {
            auto matmulOp = as<MatmulObj>(currentOp);
            std::cout << "MatMul Inputs: A=" << (matmulOp->getInputs()[0] ? matmulOp->getInputs()[0]->toString() : "nullptr")
                      << ", B=" << (matmulOp->getInputs()[1] ? matmulOp->getInputs()[1]->toString() : "nullptr") << std::endl;

            auto inputA = matmulOp->getInputs()[0];
            auto inputB = matmulOp->getInputs()[1];
            bool optimized = false; // 用于标记当前 MatMul 是否优化过

            // 检查输入 A 是否为 Transpose
            if (inputA) {
                auto inputAOp = inputA->getSource();
                if (inputAOp && inputAOp->getOpType() == OpType::Transpose) {
                    auto transposeOp = as<TransposeObj>(inputAOp);
                    auto permute = transposeOp->getPermute();
                    std::cout << "Input A is a Transpose operation: " << transposeOp->toString() << std::endl;

                    // 判断 Transpose 是否对最后两个维度的交换
                    if (permute.size() >= 2 && permute[permute.size() - 2] == static_cast<int>(permute.size() - 1) &&
                        permute[permute.size() - 1] == static_cast<int>(permute.size() - 2)) {
                        std::cout << "Found Transpose on Input A that swaps the last two dimensions." << std::endl;

                        // 更新 MatMul 的 transA 属性
                        matmulOp->setTransA(!matmulOp->getTransA());

                        // 更新连接
                        if (!transposeOp->getOutputs().empty()) {
                            auto transOutput = transposeOp->getOutputs()[0]; // 获取 Transpose 的输出张量
                            std::cout << "Changing MatMul Input A from Transpose->Input to Transpose's Source." << std::endl;

                            // 将 MatMul 的输入从 Transpose 的输出张量替换为 Transpose 的输入张量
                            auto transposeInput = transposeOp->getInputs()[0]; // 获取 Transpose 的输入张量
                            matmulOp->replaceInput(inputA, transposeInput);

                            // 删除 Transpose 输出张量的来源连接
                            transOutput->setSource(nullptr);

                            // 将 Transpose 从其输入张量的 targets 列表中移除
                            inputA->removeTarget(transposeOp);
                        }

                        // 删除 Transpose 算子和输入张量
                        std::cout << "Removing Transpose for Input A: " << transposeOp->toString() << std::endl;

                        // 确保在删除张量之前已经正确更新图结构
                        removeTensor(inputA);
                        removeOperator(transposeOp);


                        optimized = true;
                    }
                }
            }

            // 检查输入 B 是否为 Transpose
            if (inputB) {
                auto inputBOp = inputB->getSource();
                if (inputBOp && inputBOp->getOpType() == OpType::Transpose) {
                    auto transposeOp = as<TransposeObj>(inputBOp);
                    auto permute = transposeOp->getPermute();
                    std::cout << "Input B is a Transpose operation: " << transposeOp->toString() << std::endl;

                    // 判断 Transpose 是否对最后两个维度的交换
                    if (permute.size() >= 2 && permute[permute.size() - 2] == static_cast<int>(permute.size() - 1) &&
                        permute[permute.size() - 1] == static_cast<int>(permute.size() - 2)) {
                        std::cout << "Found Transpose on Input B that swaps the last two dimensions." << std::endl;

                        // 更新 MatMul 的 transB 属性
                        matmulOp->setTransB(!matmulOp->getTransB());

                        // 更新连接
                        if (!transposeOp->getOutputs().empty()) {
                            auto transOutput = transposeOp->getOutputs()[0]; // 获取 Transpose 的输出张量
                            std::cout << "Changing MatMul Input B from Transpose->Input to Transpose's Source." << std::endl;

                            // 将 MatMul 的输入从 Transpose 的输出张量替换为 Transpose 的输入张量
                            auto transposeInput = transposeOp->getInputs()[0]; // 获取 Transpose 的输入张量
                            matmulOp->replaceInput(inputB, transposeInput);

                            // 删除 Transpose 输出张量的来源连接
                            transOutput->setSource(nullptr);

                            // 将 Transpose 从其输入张量的 targets 列表中移除
                            inputA->removeTarget(transposeOp);
                        }

                        // 删除 Transpose 算子和输入张量
                        std::cout << "Removing Transpose for Input B: " << transposeOp->toString() << std::endl;

                        // 确保在删除张量之前已经正确更新图结构
                        removeTensor(inputB);
                        removeOperator(transposeOp);


                        optimized = true;
                    }
                }
            }

            // // 如果当前 MatMul 优化了，重新开始迭代，避免操作失效的迭代器
            if (optimized) {
                continue; // 跳过递增，直接处理下一个算子
            }
        }

        ++it; // 移动到下一个操作符
    }
}

Tensor GraphObj::getTensor(int fuid) const {
    for (auto tensor : tensors) {
        if (tensor->getFuid() == fuid) {
            return tensor;
        }
    }
    return nullptr;
}

void GraphObj::shape_infer() {
    for (auto &op : ops) {
        auto ans = op->inferShape();
        IT_ASSERT(ans.has_value());
        auto oldOutputs = op->getOutputs();
        IT_ASSERT(ans.value().size() == oldOutputs.size());
        // replace the old outputshape and size with new one
        for (int i = 0; i < (int)ans.value().size(); ++i) {
            auto newShape = ans.value()[i];
            auto oldShape = oldOutputs[i]->getDims();
            auto fuid = oldOutputs[i]->getFuid();
            if (newShape != oldShape) {
                auto tensor = this->getTensor(fuid);
                tensor->setShape(newShape);
            }
        }
    }
}

void GraphObj::dataMalloc() {
    // topological sorting first
    IT_ASSERT(topo_sort() == true);

    // =================================== 作业 ===================================
    // TODO：利用 allocator 给计算图分配内存
    // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
    // =================================== 作业 ===================================
    // 为每个张量分配内存

    std::vector<size_t> offsets;
    void *basePtr = nullptr;

    // 第一个循环：记录每个 tensor 的 offset
    for (size_t i = 0; i < tensors.size(); i++) {
        auto &tensor = tensors[i];
        printf("Processing tensor %zu...\n", i);

        // 分配内存
        size_t sizeInBytes = tensor->getBytes();
        printf("getBytes done ...\n");
        size_t offset = allocator.alloc(sizeInBytes * 8);
        printf("alloc done ...\n");

        if (offset == std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("Allocator failed to allocate memory");
        }

        printf("Offset allocated: %zu\n", offset);
        offsets.push_back(offset); // 记录 offset
    }
    // 获取一次 basePtr（假设所有分配都从同一 basePtr 开始）
    basePtr = allocator.getPtr();
    if (!basePtr) {
        throw std::runtime_error("Allocator returned null base pointer");
    }
    printf("BasePtr acquired: %p\n", basePtr);

    // 检查是否成功获取 basePtr
    if (!basePtr) {
        throw std::runtime_error("Base pointer is null after allocation loop");
    }

    // 第二个循环：绑定 Blob
    for (size_t i = 0; i < tensors.size(); i++) {
        auto &tensor = tensors[i];
        size_t offset = offsets[i];
        void *dataPtr = reinterpret_cast<char *>(basePtr) + offset; // 基于 offset 计算 dataPtr
        printf("Tensor %zu: BasePtr: %p, DataPtr: %p\n", i, basePtr, dataPtr);

        // 创建 Blob
        Blob blob = make_ref<BlobObj>(tensor->getRuntime(), dataPtr);
        tensor->setDataBlob(blob); // 绑定 Blob
        printf("Blob set successfully for tensor %zu\n", i);
    }

    allocator.info();
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) { return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime)); }

Tensor GraphObj::addTensor(const Tensor &tensor) {
    IT_ASSERT(tensor->getRuntime() == runtime,
              std::string("Tensor runtime mismatch: cannot add a tenosr in ") + tensor->getRuntime()->toString() + " to " + runtime->toString());
    tensors.emplace_back(tensor);
    return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
    for (auto &t : tensors)
        addTensor(t);
    return tensors;
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
    for (auto tensor : tensors) {
        IT_ASSERT(!(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));
        for (auto op : tensor->getTargets()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
        }
        auto op = tensor->getSource();
        IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
    }
    for (auto op : ops) {
        for (auto tensor : op->getInputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) != tensors.end());
        }
        for (auto tensor : op->getOutputs()) {
            IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) != tensors.end());
        }
        for (auto pre : op->getPredecessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
        }
        for (auto suc : op->getSuccessors()) {
            IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
        }
    }
    std::set<UidBaseType> s;
    // check whether two tensors with the same FUID exist
    for (auto tensor : tensors) {
        int cnt = s.count(tensor->getFuid());
        IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
        s.insert(tensor->getFuid());
    }
    return true;
}

} // namespace infini