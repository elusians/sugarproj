python XML-to-TFRecord.py -x /home/sugarproj/workspace/images/train -l /home/sugarproj/workspace/annotations/label_map.pbtxt -o /home/sugarproj/workspace/annotations/train.record

python XML-to-TFRecord.py -x /home/sugarproj/workspace/images/test -l /home/sugarproj/workspace/annotations/label_map.pbtxt -o /home/sugarproj/workspace/annotations/test.record
