#%%
def Create_mask(sz, pad):

    # # mask example test
    # att_at_v = np.ones((1, 2, 3, 4))  # (batch, n_heads, n_tokens, embed_dim)
    # att_at_v[:,:,:,-1] = 0
    # w = np.transpose(q, (0, 2, 1, 3)).reshape((1, 3, -1)) # (batch, n_tokens, n_heads x embed_dim)


    mask = torch.ones(sz)
    attn_mask = torch.ones((sz[0], sz[1], sz[1]))
    for i in range(pad.shape[0]):
        if pad[i] > 0:
            mask[i, -pad[i]:] = 0
            attn_mask[i,:,-pad[i]:] = 0
            attn_mask[i,-pad[i]:,:] = 0   # not necessary if we use a mask in the loss function
    
    '''
    # Attention Mask after softmax
    # |             |             |
    # |   Valid     |    padded   |
    # |   Tokens    |    Tokens   |
    # |             |       0     |
    # |_____________|_____________|
    # |             |             |
    # |    padded   |    padded   |
    # |    Tokens   |    Tokens   |
    # |      0      |      0      |
    # |             |             |
    '''
    return mask, attn_mask

#%%
import torch
from icecream import ic
q1 = torch.Tensor([[1, 1, 1],
                   [1, 2, 2],
                   [2, 2, 0]])
q2 = torch.Tensor([[2, 1, 1],
                   [2, 0, 1],
                   [2, 3, 3]])
segments = torch.stack((q1, q2), 0)
gts = torch.randint(0,2, (2, 3, 3))
features = torch.stack((q1+1, q2+2, q2.transpose(0,1), q1.transpose(0,1)+3, q1+1, q2+2), 0).view((2, 3, 3, 3))
print(features.shape)
print(gts.shape)
print(gts)
print('------')

max_length = 6
mix_images = False
random_tokens = False
n_tokens = torch.Tensor([3, 4]).long()
B, C, _, _ = features.shape


tokens_ids = torch.zeros((n_tokens.sum())).long()
if random_tokens:
    tokens_ids[:n_tokens[0]] = torch.randperm(n_tokens[0]).long() + 1
else:
    tokens_ids[:n_tokens[0]] = torch.arange(n_tokens[0]).long() + 1
segments[0] += 1
print(segments[0].min(), segments[0].max())

for i in range(1, B):
    if random_tokens:
        tokens_ids[n_tokens[i-1]:n_tokens[i-1]+n_tokens[i]] = torch.randperm(n_tokens[i]).long() + n_tokens[i-1] + 1
    else:
        tokens_ids[n_tokens[i-1]:n_tokens[i-1]+n_tokens[i]] = torch.arange  (n_tokens[i]).long() + n_tokens[i-1] + 1
    segments[i] += n_tokens[i-1] + 1
    print(segments[i].min(), segments[i].max())

print(tokens_ids)
print(segments)

#%%

if mix_images and random_tokens:
    ids = torch.randperm(len(tokens_ids))
    tokens_ids = tokens_ids[ids]

temp_seq, temp_lb, tokens, super_labels, pads = 5 * [torch.Tensor([])]
i_sample = 0

for i in range(len(tokens_ids)):
    pos = torch.where(segments==tokens_ids[i])
    assert len(torch.unique(pos[0])) == 1, "Tokens ids are not unique among image samples"
    
    tk = features[pos[0], :,pos[1], pos[2]].mean(0)
    lb = torch.mode(gts[pos[0], pos[1], pos[2]])[0]

    if len(temp_seq)==0:
        temp_seq = tk.unsqueeze(0)
        temp_lb = lb.unsqueeze(0)
    else:
        temp_seq = torch.cat((temp_seq, tk.unsqueeze(0)), 0)
        temp_lb = torch.cat((temp_lb, lb.unsqueeze(0)), 0)
    
    pad = 0
    if not mix_images:
        if len(temp_seq) == n_tokens[i_sample]:
            pad = max_length - len(temp_seq)
            temp_seq = torch.nn.functional.pad(temp_seq, (0, 0, 0, pad))
            temp_lb = torch.nn.functional.pad(temp_lb, (0, pad))
            i_sample += 1

    # Creating batch
    if len(temp_seq) == max_length:
        if len(tokens)==0:
            tokens = temp_seq.unsqueeze(0)
            super_labels = temp_lb.unsqueeze(0)
            pads = torch.Tensor([[pad]])
        else:
            tokens = torch.cat((tokens, temp_seq.unsqueeze(0)), 0)
            super_labels = torch.cat((super_labels, temp_lb.unsqueeze(0)), 0)
            pads = torch.cat((pads, torch.Tensor([[pad]])), 0)
        
        temp_seq, temp_lb = 2 * [torch.Tensor([])]
    
# Last sequence
if len(temp_seq):
    pad = max_length - len(temp_seq)

    temp_seq = torch.nn.functional.pad(temp_seq, (0, 0, 0, pad))
    temp_lb = torch.nn.functional.pad(temp_lb, (0, pad))

    if len(tokens)==0:
        tokens = temp_seq.unsqueeze(0)
        super_labels = temp_lb.unsqueeze(0)
        pads = torch.Tensor([[pad]])
    else:
        tokens = torch.cat((tokens, temp_seq.unsqueeze(0)), 0)
        super_labels = torch.cat((super_labels, temp_lb.unsqueeze(0)), 0)
        pads = torch.cat((pads, torch.Tensor([[pad]])), 0)

print(tokens.shape)
print(super_labels.shape)
print(pads.shape)
print(tokens_ids)
print(tokens)
print(super_labels)
print(pads)
