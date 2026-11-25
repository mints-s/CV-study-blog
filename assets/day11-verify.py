"""합성 예제 검산이다. COCO 전체 평가기를 대체하지 않는다."""
from math import isclose

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda x:(x[2]-x[0])*(x[3]-x[1])
    return inter/(area(a)+area(b)-inter)

def evaluate(pred,gt,threshold):
    used=set(); tp=0; pr=[]; rc=[]; matches=[]
    for score,box in sorted(pred,reverse=True):
        candidates=[(iou(box,g),j) for j,g in enumerate(gt) if j not in used]
        best,j=max(candidates,default=(-1,-1))
        hit=best>=threshold
        if hit: used.add(j);tp+=1
        matches.append(int(hit));pr.append(tp/len(matches));rc.append(tp/len(gt))
    ap=sum(max([p for p,r in zip(pr,rc) if r>=k/100],default=0) for k in range(101))/101
    return matches,ap

gt=[(0,0,4,4),(10,0,14,4)]
pred=[(.9,gt[0]),(.8,gt[0]),(.7,(11,0,15,4))]
assert isclose(iou(gt[1],pred[-1][1]),.6)
a,ap50=evaluate(pred,gt,.5); b,ap75=evaluate(pred,gt,.75)
assert a==[1,0,1] and b==[1,0,0]
assert isclose(ap50,253/303) and isclose(ap75,51/101)
assert isclose(evaluate([pred[0],pred[2]],gt,.5)[1],1)
print('AP50',ap50,'AP75',ap75)
for m in [[[90,0],[10,0]],[[85,5],[2,8]]]:
    ious=[m[c][c]/(sum(m[c])+sum(row[c] for row in m)-m[c][c]) for c in range(2)]
    acc=sum(m[c][c] for c in range(2))/sum(map(sum,m))
    dice=2*m[1][1]/(sum(m[1])+sum(row[1] for row in m))
    print('accuracy',acc,'IoUs',ious,'mIoU',sum(ious)/2,'FG Dice',dice)
    assert isclose(dice,2*ious[1]/(1+ious[1]))
assert isclose((85/92+8/15)/2,0.7286231884057971)
