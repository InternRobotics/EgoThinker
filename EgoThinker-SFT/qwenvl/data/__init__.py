import re

# Define placeholders for dataset paths
clevrer = {
    "annotation_path": "qwenvl/data/anno/clevrer.json",
    "data_path": "phdd2:s3://clevrer/video_train",
}

ego4dcot = {
    "annotation_path": "qwenvl/data/anno/ego4d_cot.json",
    "data_path": "cluster1:s3://videos/ego4d/videos_short320_chunked_15s",
}

ego4dlong = {
    "annotation_path": "qwenvl/data/anno/ego4d_long.json",
    "data_path": "cluster1:s3://videos/ego4d/videos_short320_chunked_15s",
}

ego4dshort = {
    "annotation_path": "qwenvl/data/anno/ego4d_short.json",
    "data_path": "cluster1:s3://videos/ego4d/videos_short320_chunked_15s",
}

egotaskqa = {
    "annotation_path": "qwenvl/data/anno/egotaskqa.json",
    "data_path": "/mnt/petrelfs/share_data/peibaoqi/egotaskqa/qa_videos",
}

egotimeqa = {
    "annotation_path": "qwenvl/data/anno/egotimeqa.json",
    "data_path": "shddnew:s3://public-dataset-snew/ego4d/clips",
}

how2short = {
    "annotation_path": "qwenvl/data/anno/howto_short.json",
    "data_path": "sssd:s3://video_pub/howto100m",
}

k400 = {
    "annotation_path": "qwenvl/data/anno/k400.json",
    "data_path": "phdd2:s3://k400",
}

nextqa = {
    "annotation_path": "qwenvl/data/anno/nextqa.json",
    "data_path": "phdd2:s3://nextqa",
}

perception = {
    "annotation_path": "qwenvl/data/anno/perception_train.json",
    "data_path": "phdd2:s3://perception/videos",
}

qaego4d = {
    "annotation_path": "qwenvl/data/anno/sft_qaego4d_mcq.json",
    "data_path": "shddnew:s3://public-dataset-snew/ego4d/clips",
}

sharegpt4o = {
    "annotation_path": "qwenvl/data/anno/sharegpt4o.json",
    "data_path": "phdd2:s3://perception/videos",
}

sharegpt4v_coco = {
    "annotation_path": "qwenvl/data/anno/sharegpt4v_coco.json",
    "data_path": "phdd2:s3://OneVision/onevision_unzip/image/sharegpt4v(coco)",
}

sharegpt4v_llava = {
    "annotation_path": "qwenvl/data/anno/sharegpt4v_llava.json",
    "data_path": "phdd2:s3://OneVision/onevision_unzip/image/sharegpt4v(llava)",
}

sharegpt4v_sam = {
    "annotation_path": "qwenvl/data/anno/sharegpt4v_sam.json",
    "data_path": "phdd2:s3://OneVision/onevision_unzip/image/sharegpt4v(sam)",
}


ssv2 = {
    "annotation_path": "qwenvl/data/anno/ssv2.json",
    "data_path": "sssd:s3://video_pub/ssv2_video",
} 

textcaps = {
    "annotation_path": "qwenvl/data/anno/textcaps.json",
    "data_path": "phdd2:s3://OneVision/onevision_unzip/image/textcaps",
} 

youcook2 = {
    "annotation_path": "qwenvl/data/anno/youcook2.json",
    "data_path": "phdd2:s3://youcook2/split_videos",
} 

videochatgpt = {
    "annotation_path": "qwenvl/data/anno/videochatgpt.json",
    "data_path": "phdd2:s3://anet/ANet_320p_fps30",
} 

visor = {
    "annotation_path": "qwenvl/data/anno/visor_1021.json",
    "data_path": "",
} 

holo_reason = {
    "annotation_path": "qwenvl/data/anno/holoassist_reasoning_sft.json",
    "data_path": "/mnt/inspurfs/HOD_t/huangyifei/HoloAssist",
} 

holo_under = {
    "annotation_path": "qwenvl/data/anno/holoassist_understanding_sft.json",
    "data_path": "/mnt/inspurfs/HOD_t/huangyifei/HoloAssist",
} 

agibot_under = {
    "annotation_path": "qwenvl/data/anno/agibot_understanding_sft.json",
    "data_path": "",
} 

agibot_reason = {
    "annotation_path": "qwenvl/data/anno/agibot_reasoning_sft.json",
    "data_path": "",
} 

object_ref = {
    "annotation_path": "qwenvl/data/anno/object_ref_340k.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
} 

region_ref = {
    "annotation_path": "qwenvl/data/anno/region_ref_320k.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
} 


egoplan = {
    "annotation_path": "qwenvl/data/anno/egoplan_train.json",
    "data_path": "",
} 

coco_300k = {
    "annotation_path": "qwenvl/data/anno/coco_300k.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
}

coco_lvis = {
    "annotation_path": "qwenvl/data/anno/coco_lvis_124k.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
}

robovqa_under = {
    "annotation_path": "qwenvl/data/anno/robovqa_understanding_sft.json",
    "data_path": "",
} 

robovqa_reason = {
    "annotation_path": "qwenvl/data/anno/robovqa_reasoning_sft.json",
    "data_path": "",
} 

gqa = {
    "annotation_path": "qwenvl/data/anno/gqa.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
} 

gqa = {
    "annotation_path": "qwenvl/data/anno/gqa.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
} 

vg = {
    "annotation_path": "qwenvl/data/anno/VG.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
} 

rec_conv = {
    "annotation_path": "qwenvl/data/anno/rec_conversation_process.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
} 

rec_detail = {
    "annotation_path": "qwenvl/data/anno/rec_detailed_description_process.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
} 

rec_region = {
    "annotation_path": "qwenvl/data/anno/rec_region_captioning_process.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
} 

refcoco = {
    "annotation_path": "qwenvl/data/anno/refcoco_1021_process.json",
    "data_path": "/mnt/inspurfs/efm_t/sys2_data/robopoint_traindata/main/images",
} 

clevr_r1 = {
    "annotation_path": "qwenvl/data/anno/clevr_r1.json",
    "data_path": "/mnt/inspurfs/HOD_t/peibaoqi/Clevr_CoGenT_TrainA_R1/images/",
} 

data_dict = {
    "clevrer": clevrer,
    "ego4dcot": ego4dcot,
    "ego4dlong": ego4dlong,
    "ego4dshort": ego4dshort,
    "egotaskqa": egotaskqa,
    "egotimeqa": egotimeqa,
    "how2short": how2short,
    "k400": k400,
    "nextqa": nextqa,
    "perception": perception,
    "qaego4d": qaego4d,
    "sharegpt4o": sharegpt4o,
    "sharegpt4v_coco": sharegpt4v_coco,
    "sharegpt4v_llava": sharegpt4v_llava,
    "sharegpt4v_sam": sharegpt4v_sam,
    "ssv2": ssv2,
    "textcaps": textcaps,
    "youcook2": youcook2,
    "videochatgpt": videochatgpt,
    "visor": visor,
    "holo_reason": holo_reason,
    "holo_under": holo_under,
    "agibot_under": agibot_under,
    "agibot_reason": agibot_reason,
    "object_ref": object_ref,
    "region_ref": region_ref,
    "egoplan": egoplan,
    "coco_300k": coco_300k,
    "coco_lvis": coco_lvis,
    "robovqa_under": robovqa_under,
    "robovqa_reason": robovqa_reason,
    "gqa": gqa,
    "vg": vg,
    "rec_conv":rec_conv,
    "rec_detail":rec_detail,
    "rec_region":rec_region,
    "refcoco":refcoco,
    "clevr_r1":clevr_r1,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []

    for dataset_name in dataset_names:

        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ['clevrer%25', 'ego4dcot', 'ego4dlong%20', 'ego4dshort', 'egotaskqa', 'egotimeqa%50', 'how2short%15', 'k400%30', 'nextqa', 'perception', 'qaego4d', 'sharegpt4o', 'sharegpt4v_coco', 'sharegpt4v_llava', 'ssv2', 'textcaps', 'youcook2', 'videochatgpt%50']
    diu = ''
    for d in dataset_names:
        diu += d
        diu += ','
    print(diu)
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
