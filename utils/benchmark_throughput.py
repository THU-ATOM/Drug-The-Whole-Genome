import os
from screening_utils import VSmodel, PocketDataset, pocket_collate_fn
import torch
import pickle
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import os
from itertools import repeat
from argparse import ArgumentParser


def one_process(gpu_index, mol_model, pocket_reps_path, batch_size=4, gpu_num=8, output_dir="output", file_id=0, gpu_offset=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index + gpu_offset)
    device = torch.device("cuda")
    model = mol_model.to(device)
    model.eval()
    if isinstance(pocket_reps_path, list):
        # for npy format
        lambda_name = lambda x: "_".join(x.split("/")[-1].strip("_").replace("_refined", "").split("_")[:-2])
        dataset = [PocketDataset(i, j, lambda_name=lambda_name) for i, j in pocket_reps_path]
        dataset = ConcatDataset(dataset) if len(dataset) > 1 else dataset[0]
    else:
        dataset = PocketDataset(pocket_reps_path)
    print(f"processing {sum([i.shape[0] for i in dataset.pocket_emb])} pockets from {pocket_reps_path}")
    sampler = DistributedSampler(
        dataset, num_replicas=gpu_num, rank=gpu_index, shuffle=False
    )
    dataLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=pocket_collate_fn,
        drop_last=False,
        num_workers=1,
    )
    pocket_name = []
    mol_score = []
    mol_index = []
    file_index = []

    if file_id > 0:
        with open(
            os.path.join(output_dir, f"merge{file_id-1}_{gpu_index}.pkl"),
            "rb",
        ) as f:
            pocket_name_merged, score_merged, ind_merged, f_ind_merged = pickle.load(f)
    else:
        score_merged, ind_merged, f_ind_merged = (
            repeat(None),
            repeat(None),
            repeat(None),
        )

    with torch.no_grad():
        if gpu_index == 0:
            dataLoader = tqdm(dataLoader)
        for (names, emb, index), score_m, ind_m, f_ind_m in zip(
            dataLoader, score_merged, ind_merged, f_ind_merged
        ):
            emb = emb.to(device)
            index = index.to(device)
            topk_score, topk_index = model(emb, index)
            pocket_name += names

            if file_id > 0:
                topk_score = torch.cat([score_m.to(device), topk_score], dim=1)
                topk_score, ind = torch.topk(
                    topk_score.contiguous(), k=100000, dim=-1, sorted=False
                )
                topk_index = torch.cat([ind_m.to(device), topk_index], dim=1)
                topk_index = topk_index[
                    torch.arange(topk_index.shape[0], device=device).reshape(-1, 1), ind
                ]
                f_ind_m = torch.cat(
                    [f_ind_m.to(device), torch.ones_like(topk_score, dtype=torch.uint8) * file_id],
                    dim=1,
                )
                f_ind_m = f_ind_m[torch.arange(f_ind_m.shape[0], device=device).reshape(-1, 1), ind]
            else:
                f_ind_m = torch.zeros_like(topk_score, dtype=torch.uint8)
            file_index.append(f_ind_m.cpu())

            mol_score.append(topk_score.cpu())
            mol_index.append(topk_index.cpu())

    if file_id > 0:
        assert pocket_name_merged == pocket_name
    with open(
        os.path.join(output_dir, f"merge{file_id}_{gpu_index}.pkl"),
        "wb",
    ) as f:
        pickle.dump((pocket_name, mol_score, mol_index, file_index), f)
    
    return None


if __name__ == "__main__":
    import torch.multiprocessing as mp
    
    parser = ArgumentParser()
    parser.add_argument("--gpu_id", "-g", type=int, default=0)
    parser.add_argument("--mol_embs", type=str, required=True)
    parser.add_argument("--zscore_embs", "-z", type=str, default=None)
    parser.add_argument("--pocket_reps", "-p", type=str, nargs=2, required=True)
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    args = parser.parse_args()

    embedding_list = [args.mol_embs,]
    gpu_num = 1
    gpu_id = args.gpu_id
    pocket_reps = [args.pocket_reps]
    batch_size = args.batch_size
    zscore_embs = args.zscore_embs if args.zscore_embs else embedding_list[0]
    output_dir = args.output_dir

    for i, f in tqdm(enumerate(embedding_list), total=len(embedding_list)):
        mol_model = VSmodel(f, zscore_embs)
        print(f"processing {mol_model.mol_embs.shape} mol embs from {f}")
        mp.spawn(
            one_process,
            args=(mol_model, pocket_reps, batch_size, gpu_num, output_dir, i, gpu_id),
            nprocs=gpu_num,
            join=True,
        )
    print(f"done, final results saved at {os.path.join(output_dir, f'merge{i}_{gpu_num-1}.pkl')}")
