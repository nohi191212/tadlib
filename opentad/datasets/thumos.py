import numpy as np
from .base import SlidingWindowDataset, PaddingDataset, filter_same_annotation
from .builder import DATASETS


@DATASETS.register_module()
class ThumosSlidingDataset(SlidingWindowDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]

        if video_anno != {}:
            # frame divided by snippet stride inside current window
            # this is only valid gt inside this window
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                window_size=self.window_size,
                # trunc window setting
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                # sliding post process setting
                fps=video_info["frame"] / video_info["duration"],
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                # training setting
                **video_anno,
            )
        )
        return results


@DATASETS.register_module()
class ThumosPaddingDataset(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
            return filter_same_annotation(annotation)

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        if video_anno != {}:
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                fps=video_info["frame"] / video_info["duration"],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
        return results


@DATASETS.register_module()
class ThumosPaddingDatasetwithIBC(PaddingDataset):
    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        gt_bc = []
        for anno in video_info["annotations"]:
            if anno["label"] == "Ambiguous":
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])
            # gt_bc
            instance_code = int(anno["instance_code"])
            # 将整数转换为至少20位的二进制字符串，前面填充0  
            binary_str = format(instance_code, '020b')  
            # 将二进制字符串转换为numpy数组，数组中的元素类型为int  
            binary_code = np.array([float(bit) for bit in binary_str], dtype=float) 

            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(anno["label"]))
                gt_bc.append(binary_code)

        if len(gt_segment) == 0:  # have no valid gt
            return None
        else:
            annotation = dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
                gt_bicodes=np.array(gt_bc, dtype=np.float32)
            )
            return self.filter_same_annotation(annotation)
        
    def filter_same_annotation(self, annotation):
        gt_segments = []
        gt_labels = []
        gt_bicodes = []
        for gt_segment, gt_label, gt_bicode in zip(annotation["gt_segments"].tolist(), \
                            annotation["gt_labels"].tolist(), annotation["gt_bicodes"]):
            if (gt_segment not in gt_segments) or (gt_label not in gt_labels):
                gt_segments.append(gt_segment)
                gt_labels.append(gt_label)
                gt_bicodes.append(gt_bicode)
            else:
                if gt_labels[gt_segments.index(gt_segment)] != gt_label:
                    gt_segments.append(gt_segment)
                    gt_labels.append(gt_label)
                    gt_bicodes.append(gt_bicode)
                else:
                    continue

        annotation = dict(
            gt_segments=np.array(gt_segments, dtype=np.float32),
            gt_labels=np.array(gt_labels, dtype=np.int32),
            gt_bicodes=np.array(gt_bicodes, dtype=np.float32),
        )
        return annotation

    def __getitem__(self, index):
        video_name, video_info, video_anno = self.data_list[index]

        if video_anno != {}:
            video_anno["gt_segments"] = video_anno["gt_segments"] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        results = self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                sample_stride=self.sample_stride,
                snippet_stride=self.snippet_stride,
                fps=video_info["frame"] / video_info["duration"],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
        
        return results