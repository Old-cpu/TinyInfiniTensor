#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        if (std::max(inputs[0]->getRank(), inputs[1]->getRank()) < 2) {
            return {};
        }
        auto A_shape = inputs[0]->getDims(), B_shape = inputs[1]->getDims();
        Shape ans = Shape(std::max(A_shape.size(), B_shape.size()));
        int a_index = A_shape.size() - 3, b_index = B_shape.size() - 3;
        for(; a_index >= 0 && b_index >= 0; --a_index, --b_index) {
            if(std::min(A_shape[a_index], B_shape[b_index]) == 1 || A_shape[a_index] == B_shape[b_index]) {
                ans[std::max(a_index, b_index)] = std::max(A_shape[a_index], B_shape[b_index]);
            } else {
                return {};
            }
        }

        for(; a_index >= 0; --a_index) {
            ans[a_index] = A_shape[a_index];
        }
        for(; b_index >= 0; -- b_index) {
            ans[b_index] = B_shape[b_index];
        }

        // get dim of last 2 element(mul).
        this->m = A_shape[A_shape.size() - 2], this->k = A_shape[A_shape.size() - 1];
        if (transA) {
            std::swap(this->m, this->k);
        }
        auto A_k = this->k;
        this->k = B_shape[B_shape.size() - 2], this->n = B_shape[B_shape.size() - 1];
        if (transB) {
            std::swap(this->k, this->n);
        }
        if (A_k != this->k) {
            return {};
        }

        ans[ans.size() - 2] = this->m;
        ans[ans.size() - 1] = this->n;


        return {{ans}};
    }

} // namespace infini