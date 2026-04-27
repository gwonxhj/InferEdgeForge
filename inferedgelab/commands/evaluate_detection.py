from __future__ import annotations

def evaluate_detection_cmd(args) -> int:
    from inferedgelab.core.detection_evaluator import (
        evaluate_detection_engine,
        save_accuracy_payload,
        save_structured_result,
    )

    eval_result = evaluate_detection_engine(
        model_path=args.model_path,
        engine=args.engine,
        engine_path=args.engine_path,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        num_classes=args.num_classes,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        iou_threshold=args.iou_threshold,
        use_rgb=args.rgb,
    )

    print(f"Evaluating detection: {args.model_path}")
    print(f"Engine          : {args.engine}")
    print(f"Images          : {args.image_dir}")
    print(f"Labels          : {args.label_dir}")
    print(f"Samples         : {eval_result['sample_count']}")
    print(f"Precision       : {eval_result['metrics']['precision']:.4f}")
    print(f"Recall          : {eval_result['metrics']['recall']:.4f}")
    print(f"F1 Score        : {eval_result['metrics']['f1_score']:.4f}")
    print(f"mAP@50          : {eval_result['metrics']['map50']:.4f}")
    print(f"mAP@50-95       : {eval_result['metrics']['map50_95']:.4f}")

    if args.out_json:
        save_accuracy_payload(args.out_json, eval_result)
        print(f"Saved accuracy  : {args.out_json}")

    if args.save_structured_result:
        result_path = save_structured_result(
            out_dir=args.out_dir,
            model_path=args.model_path,
            precision=args.precision,
            eval_result=eval_result,
        )
        print(f"Saved structured result: {result_path}")

    return 0
