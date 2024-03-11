# source code insight tips(roadmap)

- 目录结构
- 文件名
- 类名
- 类成员
- 基本数据结构
- ...


# vllm source code

```python
def _run_engine():
    # ...
    # Run the engine.
    outputs: List[RequestOutput] = []
    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()
        for output in step_outputs:
            if output.finished():
                outputs.append(output)
                if use_tqdm:
                    pbar.update(1)
    # ...
```

核心函数在于`step`生成token

```python
def step():
    # ...
    # Execute the model.
    # 最终调用Worker::execute_model
    output = self._run_workers(
        "execute_model",
        seq_group_metadata_list=seq_group_metadata_list,
        blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
        blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
        blocks_to_copy=scheduler_outputs.blocks_to_copy,
    )
    # Update the scheduler with the model outputs.
    seq_groups = self.scheduler.update(output)
    # ...
```

```python
def execute_model():
    # 换入换出所需block
    # 准备输入输出tensor
    # 调用model()
```

## 简单使用

llm engine: 不断engine step直到所有请求处理完成

```python
from vllm import EngineArgs, LLMEngine, SamplingParams

test_prompts = [
    ("To be or not to be,", SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
    ("It is only with the heart that one can see rightly", SamplingParams(n=3, best_of=3, use_beam_search=True, temperature=0.0)),
]

while True:
    # To test iteration-level scheduling, we add one request at each step.
    if test_prompts:
        prompt, sampling_params = test_prompts.pop(0)
        engine.add_request(str(request_id), prompt, sampling_params)
        request_id += 1

    request_outputs = engine.step()
    for request_output in request_outputs:
        if request_output.finished():
            print(request_output)

    if not (engine.has_unfinished_requests() or test_prompts):
        break

```

offline inference: 创建LLM, 投入prompt, 得到结果

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)
```


## 目录结构

roadmap

- [x] core/
    * 块管理
    * 调度器
- [ ] engine/
    * vllm框架顶层的交互逻辑: 推理的主循环
- [ ] entrypoints/
- [ ] model_executor/
    * vllm支持的模型
- [x] worker/
    - cache engine
    - worker
- [x] 基本数据结构抽象:
    * block
    * sequence: 一个完整的输入/输出的表示

## sequence.py

- SequenceStatu
    * enum, 任务状态标识
- SequenceData
    * 输入的token id和输出token id
- Sequence
    * seq id唯一标识
    * SequenceData
    * SequenceStatu
    * 原始prompt string
    * 输出prompt string
    * 分块记录
- SequenceGroup: **管理一个prompt生成的多个结果(Sequence)**
    * 一个请求会包含所个Sequence
    * 绑定request id和seqs
    * **flow:**
        1. `add_request`
        2. `Sequence(prompt)`封装prompt
        3. seq添加到`SequenceGroup`, 已处理一个prompt生成多个结果的情况
- SequenceGroupMetadata: **相当于handle**, 用于描述任务内容, `schedule`返回`List[SequenceGroupMetadata]`
    * `request_id: str`, `is_prompt: bool`, `seq_data: Dict[seq_id, SequenceData]`, `block_tables[Dict[seq_id, List[int]]]`
- SequenceOutputs
    * 封装输出的结果

## sampling_params.py

- SamplingParams
    * `best_of`一个prompt生成的序列的数量
    * `n`从`best_of`个中选择n个返回
    * 采样策略描述


## outputs.py

- CompletionOutput: 一个sequence的完整输出
- RequestOutput: 一个request的输出, `List[CompletionOutput]`可以含多个sequence的结果
    * 一个prompt可以生成多个sequence的结果


## block.py

- TODO: 
    * 逻辑块和物理块的关系
    * 怎么实际申请内存

- LogicalTokenBlock: 存储一段连续的token id?? 物理块信息
    * `block_number`: 一个sequence中的唯一block id
    * `token_ids`: sequence中的一段连续的token id
- PhysicalTokenBlock: 表示设备上的物理块
    * ref count

## block_manager.py

- BlockTable: `List[PhysicalTokenBlock]`维护逻辑块号到物理块号的映射
    * **一个sequence一个页表**
- BlockAllocator: 管理设备上的空闲块(物理块)
    * 只管理id是否空闲, 具体内存申请TODO
    * **AKA 管理的是分配和使用情况**
- BlockSpaceManager: 管理多个seqence的BlockTable, 即管理所有sequence的页表, **只管理页表**
    * `block_tables`: sequence到BlockTable的映射, `Dict[seq_id, BlockTable]`
    * `gpu_allocator`, `cpu_allocator`设备物理块分配器
    * `fork()`拷贝页表(BlockTable)到新sequence中
    * `swap_in(seq_group)`: CPU -> GPU
        + swap in时seq的页表是CPU页表
        + 还原GPU页表覆盖seq id的CPU页表
    * `swap_out(seq_group)`: GPU -> CPU
        + swap out时seq的页表是GPU页表
        + swap out前需要判断sequence是否已完成(GPU块是否正在使用)
        + 还原CPU页表覆盖seq id的GPU页表
    * TODO:
        + append_slot: **COW**


## cache_engine.py

- 设备真正的内存申请和释放疑似就是这里, 会调用csrc
    * 初始的申请在哪?

- **申请大块tensor作为cache空间**: `allocate_gpu_cache`, `allocate_cpu_cache`
    * 以layer为单位(`gpu_cache: List[KVCache]`), 申请大块的tensor(num block * block shape)作为cache
- swap in/swap out: 使用block id索引cache完成换入换出
    * 每层layer都要执行换入换出
    * 通过cache的指针 + offset索引block数据位置
    * 使用cudaMemcpyAsync拷贝
- copy
    + 使用pytorch API来简化显存申请释放的流程, 然后简单的`value[] = value[]`
    * QA:
        + **copy是copy什么**
            + 对一个block添加token时发现block有共用, 此时触发copy on write。得到旧block的src和新block的dst
        + 处理换入换出为什么还有copy
            + 为了处理COW的情况
        * 怎么拷贝的
            + 简单的`value[] = value[]`



## scheduler.py

> 调度器管理和返回需要操作的block的id, 具体的换入换出传递到执行器执行

- PreemptionMode
- SchedulerOutputs: dataclass, 封装调度的结果, 换入换出了哪些块
- Scheduler
    * `block_manager`: 所有页表的管理器
    * runing队列: 当前step要执行的seq
    * swapped队列: seq执行到一半被换出的, 优先级高于waiting
    * waiting队列: 用户输入的prompt还没轮到执行
    * **flow**
        + 每次step调用一下`schedule()`, 获取要**执行的seq和要操作的block**
        + `schedule() -> _schedule()`更新running队列, 尽可能用满, 有空闲时就可以加载swapped的和waiting的seq
        + `schedule()`返回要执行的sequence和要操作的block


## worker.py

- `profile_num_available_blocks()`
    * 探测最大可以用来做cache的block数
- `_prepare_inputs()`
    * 将seq的索引组织成平坦的数据, 以便传入模型
    * 对于新prompt, group中所有seq一样, 插入首个seq的完整的token
    * 对于decoding, 多个seq产生了各自token, 插入各个seq的最新token(最后一个token)
- `execute_model()`: 执行器
    1. 操作要操作的block(换入, 换出, 拷贝)
    2. `prepare_input`, 将数据平坦化以便传入model执行
    3. 执行model


## swap in/swap out/copy细节

- cache engine
- block manager

TODO:

## 数据流

> 学习vllm是如何设计的

- prompt通过`add_request`传入
    * prompt抽象成sequence传入调度器
- 启动推理: 不断的step生成结果, 直到所有请求处理完成
- step
    * 同schedule获取一组当前要执行的sequence和要操作的block:
        + 更新running队列: 调度新sequence来执行
            1. 没有空闲块时要替换sequence, 不断swap out直到可以容纳新sequence, 填充要swap的block
            2. 新running队列插入新sequence, 填充需要拷贝的block
            3. 更新当前running队列为新running队列
        + 更新swapped队列: 尽可能填满running队列, 有空闲时把之前换出的seq也换入执行
            + 记录要换入的block和要copy的block
        + 更新waiting队列: 如果swapped的seq加入后仍有空闲, 加载waiting的seq, 分配资源
            + Q: 加载waiting的seq在哪申请的资源? `model().cuda()`
        + **遍历running队列**, 整理出当前step要执行的seq, 和需要操作的block
    * run worker: 执行seq, 操作block
        + 真正操作block, 该换入的换入, 该换出的换出
            + 具体的换入换出操作细节在其他章节讨论
        + ⭐**`_prepare_inputs`**
            + 将新prompt的token和decoding的token合并平铺在一起
            + 一个新prompt对应一个seq group中所有的seq是一样的, 把其所有token插入input即可
            + 正在decoding的seq group各自生成了新的的token, 将各个seq中的最新token加入input即可
    * 更新schedule: `update()`
        + 处理beam search
            + TODO
        + output的token追加到seq的结果token中
        + TODO: 情况比较复杂, 涉及fork和beam search






