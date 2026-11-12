"""Thin external verl loader for JSON-encoded conversations/tool schemas.

This module is imported only inside a user's independently installed verl.
"""

import json
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset


class WorldlineSFTDataset(MultiTurnSFTDataset):
    def _read_files_and_process(self):
        super()._read_files_and_process()
        for key in (self.messages_key, self.tools_key):
            if key in self.dataframe:
                self.dataframe[key] = self.dataframe[key].map(
                    lambda value: json.loads(value) if isinstance(value, str) else value
                )
        self.messages = self.dataframe[self.messages_key].tolist()
        self.tools = self.dataframe[self.tools_key].tolist() if self.tools_key in self.dataframe else None
        if self.image_key in self.dataframe and all(len(images) == 0 for images in self.dataframe[self.image_key]):
            self.dataframe = self.dataframe.drop(columns=[self.image_key])
