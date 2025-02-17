#include "core/graph.h"
#include "operators/transpose.h"
#include "operators/matmul.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
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

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来实现指定的图优化规则
        // 图优化规则如下：
        // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
        // 2. 合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，就可以将transpose融入到矩阵乘算子的属性中去）
        // =================================== 作业 ===================================
        IT_ASSERT(topo_sort() == true);

        auto ops_len = ops.size();
        for(size_t i = 0; i < ops_len; ++i){
            auto op = ops[i];
            if (op->getOpType() == OpType::Transpose) {
                // transpose 自己input/output size只可能为1
                auto op_output = op->getOutput(), op_input = op->getInputs()[0];
                auto now_transpose = as<TransposeObj>(op);
                auto now_permute = now_transpose->getPermute();
                auto succs = op_output->getTargets();

                // check for 不能优化的部分.这里简化了optimize,只有全删和全不删.
                bool delete_ok = true;
                for(auto& succ : succs){
                    // condition 1: all transpose
                    if(succ->getOpType() != OpType::Transpose){
                        delete_ok = false;
                        break;
                    }
                    // condition 2: permute is same.
                    auto succ_transpose = as<TransposeObj>(succ);
                    auto succ_permute = succ_transpose->getPermute();
                    for(size_t i = 0; i < now_permute.size(); ++i){
                        if(succ_permute[i] != now_permute[i]) {
                            delete_ok = false;
                            break;
                        }
                    }
                    if(!delete_ok){
                        break;
                    }
                }
                if (!delete_ok){
                    continue;
                }

                // 更新graph结构.
                // 感觉自己设计错了,该写prev的.
                for(auto& succ : succs){
                    auto succ_output = succ->getOutput();
                    auto succ_succ_objs = succ->getSuccessors();

                    for(auto& succ_succ_obj : succ_succ_objs){
                        succ_succ_obj->replaceInput(succ_output, op_input);
                        op_input->addTarget(succ_succ_obj);
                    }
                    this->removeTensor(succ_output);
                }

                for(auto& pred : op->getPredecessors()){

                    pred->removeSuccessors(op);
                }

                for(auto& succ : succs){
                    for(auto& succ_succ : succ->getSuccessors()){
                        succ_succ->removePredecessors(succ);
                    }
                }
                op_input->removeTarget(op);
                this->removeOperator(op);
                --ops_len;

                for(auto& succ : succs){
                    this->removeOperator(succ);
                    --ops_len;
                }
                this->removeTensor(op_output);
            } else if (op->getOpType() == OpType::MatMul) {
                auto matmul_op = as<MatmulObj>(op);
                for(auto prev_op : op->getPredecessors()){
                    if(prev_op->getOpType() == OpType::Transpose && prev_op->getSuccessors().size() == 1){
                        auto trans_op = as<TransposeObj>(prev_op);
                        auto perm = trans_op->getPermute();
                        auto delete_ok = true;
                        for(size_t i = 0; i < perm.size() - 2; ++i){
                            if(perm[i] != int(i)){
                                delete_ok = false;
                                break;
                            }
                        }
                        if(!delete_ok || (perm[perm.size() - 1] != int(perm.size() - 2)) 
                            || (perm[perm.size() - 2] != int(perm.size() - 1))){
                            continue;
                        }
                        
                        // 更新graph结构.
                        auto trans_input = trans_op->getInputs()[0], trans_output = trans_op->getOutput();

                        if(trans_output->getGuid() == matmul_op->getInputs()[0]->getGuid()){
                            matmul_op->setTransA(!matmul_op->getTransA());
                        } else if (trans_output->getGuid() == matmul_op->getInputs()[1]->getGuid()){
                            matmul_op->setTransB(!matmul_op->getTransB());
                        }
                        
                        op->removePredecessors(trans_op);
                        for(auto & prev_trans : trans_op->getPredecessors()) {
                            prev_trans->removeSuccessors(trans_op);
                            prev_trans->addSuccessors(matmul_op);
                            op->addPredecessors(prev_trans);
                        }
                        
                        trans_input->addTarget(matmul_op);
                        trans_input->removeTarget(trans_op);
                        
                        matmul_op->replaceInput(trans_output, trans_input);
                        trans_output->removeTarget(matmul_op);

                        this->removeOperator(trans_op);
                        this->removeTensor(trans_output);

                        --i, --ops_len;
                    }
                }
            }
        }
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        vector<size_t> offsets;
        for(auto &ts : tensors){
            offsets.push_back(allocator.alloc(ts->getBytes()));
        }

        auto ptr = allocator.getPtr();
        for(size_t i = 0; i < tensors.size(); ++i){
            auto blob = make_ref<BlobObj>(runtime, ptr + offsets[i]);
            tensors[i]->setDataBlob(blob);
        }
            
        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini