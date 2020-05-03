#ifndef PROJECT_KMEANS_RECORD_H
#define PROJECT_KMEANS_RECORD_H

#include <math.h>

class Record{

private:
    int id;
    double px;
    double py;
public:
    Record(int id, double px, double py){
        this->id =id;
        this->px=px;
        this->py = py;
    }
    Record(Record const &r){
        this->id = r.id;
        this->px = r.px;
        this->py = r.py;
    }
    Record(){
        this->id = -1;
        this->px = 0;
        this->py = 0;
    }
    int getId() const {
        return id;
    }
    double getPx() const {
        return px;
    }
    double getPy() const {
        return py;
    }
    void setId(int tid){
        id = tid;
    }
};

bool operator<(const Record & one, const Record & two){
    return (one.getId() < two.getId());
}

float record_distance(Record r1, Record r2){
    return sqrt(pow((r1.getPx()-r2.getPx()),2)+pow((r1.getPy()-r2.getPy()),2));
}

#endif //PROJECT_KMEANS_RECORD_H
