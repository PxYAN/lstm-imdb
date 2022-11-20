from copy import deepcopy
import csv
import pymysql


def getuserlist(taskid, bizid):
    db = pymysql.connect(host='39.103.169.22',
                         user='admin',
                         password='admin@2021',
                         database='edusoho',
                         charset='utf8')
    print("数据库连接成功！")
    # try:

    cursor = db.cursor()
    sql = "select * from events where task_id = %s group by user_id"
    cursor.execute(sql, taskid)
    result = cursor.fetchall()
    userlist = []
    for data in result:
        userlist.append(data[5])

    userlist1 = deepcopy(userlist)  # list中的.remove()没有把符合条件的元素删除干净
    for userid in userlist:
        # userid = 112
        sql = "select min(score) from biz_answer_report where assessment_id = %s and user_id = %s"
        cursor.execute(sql, (bizid, userid))
        examresult = cursor.fetchall()
        if str(examresult[0][0]) == 'None' or str(examresult[0]) == 'None':
            userlist1.remove(userid)
            continue
        else:
            sql1 = "select current from events where task_id = %s and user_id = %s"
            cursor.execute(sql1, (taskid, userid))
            resultuser = cursor.fetchall()
            if (len(resultuser) <= 1):
                userlist1.remove(userid)
                continue

    # print(userlist1,len(userlist1))

    return userlist1
    # except Exception as e:
    #     print("查询失败", e)


def getlenofcourse(taskid):
    db = pymysql.connect(host='39.103.169.22',
                         user='admin',
                         password='admin@2021',
                         database='edusoho',
                         charset='utf8')
    cursor = db.cursor()

    sql1 = "select length from course_task where id = %s"
    cursor.execute(sql1, taskid)
    resultuser = cursor.fetchall()
    return resultuser[0][0]


def getseq(taskid, userid):
    db = pymysql.connect(host='39.103.169.22',
                         user='admin',
                         password='admin@2021',
                         database='edusoho',
                         charset='utf8')
    cursor = db.cursor()

    sql1 = "select type from events where task_id = %s and user_id = %s"
    cursor.execute(sql1, (taskid, userid))
    seq = list()
    resultuser = cursor.fetchall()
    for i in range(len(resultuser)):
        print(resultuser[i][0])
        seq.append(int(resultuser[i][0]))
    print("\n")
    write(seq)





def write(seq):
    csvFile = open("clickseq-balance.csv", 'a+', newline='')

    # try:
    writer = csv.writer(csvFile)
    writer.writerow(seq)


def getdata():
    # video_list = [68, 70, 106, 75, 95, 76, 77, 78, 80, 91, 96, 98, 99, 83, 97, 100, 101, 84, 85, 86, 87, 88, 89, 90]
    # biz_list = [21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    video_list = [68, 70, 106, 75, 95, 76, 77]
    biz_list = [21, 22, 23, 25, 26, 27, 28]
    userlist = []
    max = 0
    count = 0
    for i in range(len(video_list)):
        userlist = getuserlist(video_list[i], biz_list[i])
        # userlist = Allgetalluserlist.getuserlist(68,21)
        # length = getlenofcourse(video_list[i])
        # print(len(userlist))

        for userid in userlist:
            getseq(video_list[i], userid)
            count += 1
            print("第", count)
            # Note = open('/Data/time.txt', mode='a')
            # j = 1
            # while j <= 12:
            #     print(j)
            #     Note.write(str(j) + '\n')
            #     j += 1
    # Note.close()

if __name__ == '__main__':
    getdata()