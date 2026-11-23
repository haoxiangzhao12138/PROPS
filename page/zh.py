# Chinese text of the project page, shown by the language switch. The English text is in make_page.py, and every
# entry here translates the entry of the same name there.
TITLE = "视觉生成器能否保持道具的当前外观？"
NAV = {"overview": "概览", "twins": "孪生历史", "train": "PROPS-Train", "leaderboard": "榜单", "findings": "主要发现",
       "evaluate": "评测", "citation": "引用"}
BTN = {"paper": "论文 (PDF)", "arxiv": "arXiv", "code": "代码", "bench": "PROPS-Benchmark", "train": "PROPS-Train"}
HEAD = {"abstract": "摘要", "twins": "孪生历史", "train": "PROPS-Train", "leaderboard": "榜单", "findings": "主要发现",
        "evaluate": "评测你的模型", "citation": "引用"}
CAP = {
    "overview": "PROPS 与 PROPS-Train。左：按要求和子类统计的对数，以及每种要求下两个对中道具的两种外观，其中 Reuse 的裁图"
                "取自 SenseNova-U1.5 所画的帧。右：按轨道统计的链数，其中交切链按切走的镜头数细分，合成链按变化类型细分；"
                "以及每个轨道中一条链的关键帧，并标出其监督目标。",
    "method": "孪生历史。两条孪生历史的脚本完全相同，只有道具的外观不同，所以只凭脚本画出的同一个答案不可能同时对上两侧。"
              "下方：八个子类中孪生 A 的历史。",
    "pipeline": "在一个 Recall-Start 对上构建 PROPS。一次编辑调用同时改变证据帧和查询帧里的道具，每种外观各做一次。判官必须在两条"
                "孪生历史的真值查询帧上都选对外观。每一对按一个子类排布。",
    "train": "PROPS-Train：每个轨道的链数和监督目标数、每类变化的合成链数，以及交切链中离开一条故事线到回到它之间的镜头数。",
    "agents": "同一条 Resist-Stale 历史经过两种 agent 流水线。闭源规划器把证据帧交给编辑器，并写明要保留的外观。",
    "qual": "phase A 上的示例。",
}
GROUP = {"Recall": "有效外观出现在好几个镜头之前",
         "Resist": "一个过时的外观、另一个地点的外观或相似物体上的外观，与有效外观相竞争",
         "Reuse": "有效外观是模型自己在历史中较早画出的"}
WHERE = {
    "Recall-Start": "在第 0 帧",
    "Recall-Mid": "在中间的一个关键帧",
    "Recall-Far": "在第 0 帧，中间插入了来自其他链的帧，使它最远在八个镜头之前",
    "Resist-Stale": "由后来的一条脚本给出，它说道具已被替换，而第 0 帧显示的是另一侧的外观",
    "Resist-Intercut": "在本链中，本链与另一段视频交替出现，那段视频里常有同类物体",
    "Resist-Lookalike": "在本链中，插入的一帧里有一个同类物体，呈现的是另一侧的外观",
    "Reuse-Own": "在模型自己画的证据帧里，脚本只提到这个外观一次",
    "Reuse-Replaced": "在模型自己画出的替换物里，脚本写明了新外观",
}
TRACK = {"synth": "在时间、空间或因果上有二到五处变化的合成链",
         "intercut": "在链 B 的一到六个镜头之后回到链 A",
         "edit": "每步一处脚本写明的改动，其余一切保持不变",
         "real": "真实视频中镜头回到先前的地点"}
FAMILY = {"umm": "统一多模态模型", "image": "图像生成与编辑模型", "agent": "Agent（开源规划器 + 编辑器）",
          "video": "关键帧条件视频模型", "closed": "闭源图像生成器", "closed_agent": "使用闭源规划器或编辑器的 Agent",
          "small_agent": "使用较小开源规划器或其他开源编辑器的 Agent", "train": "在 PROPS-Train 上微调"}
REF = {"Ground-truth frames": "真值帧", "Copy frame 0": "复制第 0 帧", "Copy previous frame": "复制前一帧"}
METRIC = {
    "Overall": ("Recall、Resist 和 Reuse 的均值。一对只有在两侧都呈现各自的道具外观并完成请求时才通过。",
                "Overall 通过率，对的百分比"),
    "Follow": ("输出呈现自己那一侧道具外观的侧所占比例。诊断指标，不计入 Overall。", "Follow：呈现自身外观，侧的百分比"),
    "Exec": ("输出完成了脚本请求的侧所占比例。诊断指标，不计入 Overall。", "Exec：完成请求，侧的百分比"),
}
GROUP_DESC = "{subs} 的均值：{what}。"
SUB_DESC = "{n} 对。有效外观{where}。"
AXIS = "{k}，通过的对的百分比"
PAIRS = "{n} 对"
VALID = "有效外观{where}。"
TRACK_TIP = "<b>{t}</b>：{what}<br>{v}K 个目标，占训练集的 {pct}%"
TEXT = {
    "twins": "每一对有两条历史，它们以同一个请求结尾，但同一个道具呈现两种不同的外观，所以每种外观在一侧是对的、在另一侧是错的。"
             "只有两个输出都呈现各自的外观并完成请求，这一对才通过。这 {pairs} 对分属三种要求、八个子类。",
    "train": "80,250 条训练链给出四个轨道中的 120,339 个监督目标。每个目标都要带回先前的状态，或把脚本写明的改动延续下去，"
             "所以无法靠复制前一帧画出来。训练数据与基准不共用任何来源视频组。",
    "full": "完整基准：{n} 对，开源系统",
    "phase_a": "Phase A：{n} 对，全部系统",
    "lb_note": "对的通过率，单位 %。子类分数是该子类中通过的对所占的百分比，Recall、Resist 和 Reuse 是各自子类的均值，Overall 是"
               "这三者的均值。Overall 后面的小条表示它在来源视频簇上做 bootstrap 得到的 95% 区间。Follow 是呈现自身外观的侧所占比例，"
               "Exec 是完成请求的侧所占比例。† 表示闭源模型或闭源规划器，闭源系统只在 phase A 上运行。粉色名称是在 PROPS-Train 上"
               "微调的模型，Overall 后面的小数字是相对基础模型的提升。鼠标悬停在单元格上可以看该系统的详细分数，点击列名可以按该列"
               "排序。如需添加一行，请在代码仓库提 issue。",
    "scatter": "phase A 上的 Overall 与 Exec",
    "sc_note": "phase A 的 {n} 对上的全部系统，开源和闭源系统在这些对上可以直接比较。大多数系统完成请求的侧远多于它们通过的对："
               "它们画出了另一侧的外观，或两种外观都没画出。阴影区域分别圈出开源系统和闭源系统，星号表示真值帧，叉号表示复制第 0 帧"
               "或复制前一帧，箭头从基础模型指向它在 PROPS-Train 上的微调版本。鼠标悬停在点上可以看分数。",
    "eval_note": "代码仓库的 README 介绍了每一步、两个打分模型和各个划分。",
}
FIND = [
    ("Prop drift 普遍存在。",
     "未经微调的最强开源系统 FLUX.3-klein 和 SenseNova-U1.5 在完整基准上分别通过 {f3}% 和 {sn}% 的对。phase A 上最强的系统"
     " GPT-5.6 + GPT Image 2.5 通过 {top}%。"),
    ("完成请求不等于保持外观。",
     "能完成大多数请求的系统，常常画出另一侧的外观或两种外观都没画出，因此通过的对要少得多。在 Overall 对 Exec 的散点图里，"
     "大多数系统完成请求的侧远多于它们通过的对。"),
    ("Agent 同时受理解和生成两方面的限制。",
     "以 Qwen-Image-Edit 为编辑器时，换成闭源规划器可使 agent 在 phase A 上从 {q38}% 升到 {q38max}%，而换用 GPT Image 2.5 可使"
     " GPT-5.6 agent 从 {g56}% 升到 {g56gi}%。"),
    ("PROPS-Train 减轻了 prop drift。",
     "在完整基准上，微调使 SenseNova-U1.5 从 {sn}% 升到 {snft}%，使 BAGEL 从 {bg}% 升到 {bgft}%，LoRA 微调同样提升了"
     " Qwen-Image-2.1 和 Qwen-Image-Edit。"),
]
# strings that app.js draws, in both languages
UI = {
    "en": {"system": "System", "type": "Type", "mean": "Mean", "diag": "Diagnostics",
           "diag_tip": "Rates over sides, not part of Overall", "ci": "95% interval of Overall: {a} to {b}",
           "of": "{k} of {b}: {v} ({d})", "open": "{n} open systems", "closed": "{n} closed systems †",
           "range": "pass {a} to {b}%", "gt": "ground truth",
           "legend": [["unified model", "image generator", "agent", "video model"],
                      ["open", "closed †", "fine-tuned on PROPS-Train"],
                      ["ground truth", "copying a frame", "base model to its fine-tune"]],
           "best": "Best: {f} {fv}% on the full benchmark, {a} {av}% on phase A", "copied": "Copied"},
    "zh": {"system": "系统", "type": "类型", "mean": "均值", "diag": "诊断指标",
           "diag_tip": "按侧计算的比率，不计入 Overall", "ci": "Overall 的 95% 区间：{a} 到 {b}",
           "of": "{b} 的 {k}：{v}（{d}）", "open": "{n} 个开源系统", "closed": "{n} 个闭源系统 †",
           "range": "通过率 {a} 到 {b}%", "gt": "真值",
           "legend": [["统一多模态模型", "图像生成器", "agent", "视频模型"],
                      ["开源", "闭源 †", "在 PROPS-Train 上微调"],
                      ["真值", "复制一帧", "基础模型指向其微调版本"]],
           "best": "最高：完整基准上 {f} {fv}%，phase A 上 {a} {av}%", "copied": "已复制"},
}
