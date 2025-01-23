#include "core/graph.h"
#include "core/runtime.h"
#include "operators/concat.h"

#include "test.h"

namespace infini {

TEST(Concat, NativeCpu) {
    Runtime runtime = NativeCpuRuntimeObj::getInstance();
    Graph g = make_ref<GraphObj>(runtime);

    auto t1 = g->addTensor({2, 2, 3, 1}, DataType::Float32);
    auto t2 = g->addTensor({2, 2, 1, 1}, DataType::Float32);
    auto t3 = g->addTensor({2, 2, 2, 1}, DataType::Float32);
    auto op = g->addOp<ConcatObj>(TensorVec{t1, t2, t3}, nullptr, 2);
    g->dataMalloc();
    t1->setData(IncrementalGenerator());
    t2->setData(OneGenerator());
    t3->setData(OneGenerator());

// 	std::vector<float> dataT1 = t1->getData(); // 假设有获取数据的方法
// std::vector<float> dataT2 = t2->getData();
// std::vector<float> dataT3 = t3->getData();
	std::cout << "Data for t1: " << t1->toString() << std::endl;
	std::cout << "Data for t2: " << t2->toString()  << std::endl;
	std::cout << "Data for t3: " << t3->toString()  << std::endl;
	std::cout << "Data for t4: " << op->getOutput()->toString()  << std::endl;

    runtime->run(g);
	printf("out:%s",op->getOutput()->toString().c_str());
    EXPECT_TRUE(op->getOutput()->equalData(
        vector<float>{0, 1, 2, 1, 1, 1, 3, 4,  5,  1, 1, 1,
                      6, 7, 8, 1, 1, 1, 9, 10, 11, 1, 1, 1}));
}

} // namespace infini
