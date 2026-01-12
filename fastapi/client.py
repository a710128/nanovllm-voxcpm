import aiohttp
import asyncio
import numpy as np
import soundfile as sf

async def tts_request(text : str):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://localhost:8000/generate", json={"target_text": text}) as response:
            return await response.content.read()

async def main():
    texts = [
        "地方志都没有，就凭脑子记，有些不是那么准确的，有些个就遗忘。这就随时代呀，都都在改变。嗯，这是很正常的事。",
        "有这么一个人呐，一个字都不认识，连他自己的名字都不会写，他上京赶考去了。哎，到那儿还就中了，不但中了，而且升来升去呀，还入阁拜相，你说这不是瞎说吗？哪有这个事啊。当然现在是没有这个事，现在你不能替人民办事，人民也不选举你呀！我说这个事情啊，是明朝的这么一段事情。因为在那个社会啊，甭管你有才学没才学，有学问没学问，你有钱没有？有钱，就能做官，捐个官做。说有势力，也能做官。也没钱也没势力，碰上啦，用上这假势力，也能做官，什么叫“假势力”呀，它因为在那个社会呀，那些个做官的人，都怀着一肚子鬼胎，都是这个拍上欺下，疑神疑鬼，你害怕我，我害怕你，互相害怕，这里头就有矛盾啦。由打这个呢，造成很多可笑的事情。今天我说的这段就这么回事。",
        "在很久很久以前，有一个国王。他把他的国家治理的非常好，国家不大，但百姓们丰衣足食，安居乐业，十分幸福。国王有三位美丽可爱的小公主，三位小公主们从生下来就具有一种神奇的魔力，当她们哭泣的时候，落下的眼泪会化作一颗颗晶莹剔透的钻石，价值连城。",
        "近日，陕西多地遭遇高温天气。7月15日，全省有8个气象站最高气温突破历史极值，多地发布高温红色预警。16日，多地高温持续，西安、宝鸡、咸阳、渭南、汉中、安康等地达40℃以上。陕西省气象台预计，从17日开始，部分区域将出现分散性降雨，持续多日的高温晴热有望得到缓解。",
        "Some days the sunsets would be purple and pink. And some days they were a blazing orange setting fire to the clouds on the horizon. It was during one of those sunsets that my father's idea of the whole being greater than the sum of its parts moved from my head to my heart.",
        "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character. I have a dream today! I have a dream that one day, down in Alabama, with its vicious racists, with its governor having his lips dripping with the words of interposition and nullification; one day right down in Alabama little black boys and black girls will be able to join hands with little white boys and white girls as sisters and brothers. I have a dream today! I have a dream that one day every valley shall be exalted, and every hill and mountain shall be made low, the rough places will be made plain, and the crooked places will be made straight, and the glory of the Lord shall be revealed and all flesh shall see it together.",
        "MiniCPM4: Ultra-Efficient LLMs on End Devices. This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose CPM.cu that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Sufficient evaluation results show that MiniCPM4 outperforms open-source models of similar size across multiple benchmarks, highlighting both its efficiency and effectiveness. Notably, MiniCPM4-8B demonstrates significant speed improvements over Qwen3-8B when processing long sequences. Through further adaptation, MiniCPM4 successfully powers diverse applications, including trustworthy survey generation and tool use with model context protocol, clearly showcasing its broad usability. In the future, we will continually investigate efficient training and inference of LLMs. In terms of model architecture, we will devote our effort to the efficient sparse model architecture with the target to enable LLMs process infinitely long sequences on end-side devices. In terms of data construction, our research will focus on improving the quality of existing corpus and synthesise large-scale reasoning-intensive pre-training datasets, which can significantly improve the foundational capabilities of LLMs. In addition, we will continually explore the great potential of reinforcement learning to enable LLMs to learn skills from various environments. As for the inference systems, we plan to develop efficient systems for most end-side platforms, which can help the community to run and evaluate our model.",
        "陈昼的指尖在控制台上游走，汗水在防护服内侧凝结成细小的水珠。全息投影里，那道幽蓝色的时空褶皱正在以肉眼可见的速度扩张，如同某种活物的呼吸。“它在吞噬锚点。” 女人的声音突然在背后响起，陈昼猛地转身，看见她正徒手触摸着能量屏障。淡紫色的电弧在她指缝间跳跃，那些本应撕裂钢铁的能量流却温顺得像溪流。“你到底是谁？” 陈昼的掌心按在紧急制动按钮上，这个按钮连接着时空枢纽的自毁程序。三天前他在例行巡检时发现了这个不速之客，她没有携带任何通行标识，却能自由出入需要七重权限的核心区域。女人缓缓转过身，兜帽滑落的瞬间，陈昼的瞳孔骤然收缩。那张脸与他钱包里夹层照片上的姑娘一模一样 —— 那是五年前在时空风暴中失踪的搭档林夏。“我是林夏，但不是你的林夏。” 她抬起左手，手腕上的时空坐标仪显示着一串混乱的数字，“来自第 734 号平行宇宙，那里的枢纽已经坍缩了。”​控制台突然发出刺耳的警报，全息投影里的褶皱中心浮现出旋转的星云状物质。陈昼注意到那些物质正在剥离金属表面的分子，观测台的合金墙壁以每秒三厘米的速度变得透明。“它不是自然形成的。” 林夏的指尖划过数据屏，调出一组加密文件，“每个宇宙的枢纽都在同时出现褶皱，像是有人在编织一张跨维度的网。”​陈昼的目光落在文件末尾的签名上，那串扭曲的符号与他昨夜在维修日志上发现的划痕完全一致。三天前首次发现异常时，他在冷却管道的内壁看到过同样的印记，当时以为是机械磨损的痕迹。“这些符号……”​“时空蛀虫的标记。” 林夏的声音带着金属摩擦般的质感，“它们以时间流为食，已经蛀空了十三个平行宇宙。”​观测台的灯光开始诡异地闪烁，陈昼的腕表指针突然逆时针疯狂旋转。他看见自己的手掌上浮现出细密的裂纹，仿佛即将碎裂的玻璃。“你的时间线正在崩解。” 林夏按住他的肩膀，她的指尖传来冰冷的触感，“每个接触过褶皱的人都会这样，这是被蛀虫标记的征兆。”​陈昼突然想起三天前的细节：他在检查能量导管时，曾被一股无形的力量拖拽，左腕留下过环形的瘀青。当时医疗机器人诊断为机械压迫，现在看来那分明是某种生物咬痕。“734 号宇宙的我，也是这样开始的吗？” 他盯着林夏的眼睛，试图从那双与记忆中完全相同的眸子里找到破绽。",
    ] * 5
    tasks = set()

    cnt = 0
    task_idx = 0
    while task_idx < len(texts) or len(tasks) > 0:
        while len(tasks) < 10 and task_idx < len(texts):
            tasks.add(asyncio.create_task(tts_request(texts[task_idx])))
            task_idx += 1
        
        done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if cnt < 10:
                wav = np.frombuffer(task.result(), dtype=np.float32)
                sf.write(f"test_{cnt}.wav", wav, 44100)
            cnt += 1
            print(f"Processed {cnt} tasks")

if __name__ == "__main__":
    asyncio.run(main())
