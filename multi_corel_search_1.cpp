#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <random>
#include <limits>
#include <algorithm>
#include <set>
#include <sstream>
#include <numeric>
#include <chrono>

using namespace std;

#define parameter_k 10
#define epsilon 1
#define max_line 1000

vector<vector<double>> read_embeddings(const string& filepath) {
    vector<vector<double>> data; // データを格納する2次元ベクトル

    ifstream file(filepath);
    if (!file.is_open()) {
        cerr << "ファイルを開けませんでした。" << endl;
        return data; // ファイルが開けない場合、空のベクトルを返す
    }

    string line;
    int count_line = 0;

    while (getline(file, line) && count_line < max_line) {
        stringstream ss(line);
        vector<double> row; // 1行のデータを格納するベクトル
        string token;

        while (getline(ss, token, ',')) {
            double value = stod(token); // カンマで区切られた値を浮動小数点数に変換
            row.push_back(value); // 行の各値をベクトルに追加
        }

        data.push_back(row); // 行のデータを2次元ベクトルに追加
        count_line++;
    }

    file.close();
    return data;
}




struct Neighbor_v{      //構造体（要素番号と距離）
    int num;
    double dist;
    bool operator<(const Neighbor_v& other) const {
        return dist > other.dist;       //distの小ささで比較
    }
};

struct density{      //構造体（要素番号と密度）
    int num;
    double dent;
};


struct info{      //構造体（要素番号と密度）
    int num;
    int graph;
    double dent;
    double dist;
    bool operator<(const info& other) const {
        return dent < other.dent;       //dentの大きさで比較
    }
};


int random(int high, int mynum){       //0からhighまででmynumとはかぶらない乱数を発生
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0,high);
    int num_random = dist(gen);
    while (num_random == mynum) {
        num_random = dist(gen);
    }

    return num_random;
}

vector<Neighbor_v> sample(int k, int n, int mynum){        //重複なしでmynumとかぶらない乱数をk-1までからn個発生
    vector<Neighbor_v> random_vec;
    double inf = numeric_limits<double>::infinity();
    set<int> random_set;
    while (random_set.size() < n) {
        Neighbor_v neighbor;
        neighbor.num = random(k - 1, mynum);
        
        // 重複しない場合のみ結果に追加
        if (random_set.find(neighbor.num) == random_set.end() && neighbor.num != mynum) {
            random_set.insert(neighbor.num);
            neighbor.dist = inf;          //変更要素あり
            random_vec.emplace_back(neighbor);
        }
    }
    return random_vec;
}

vector<vector<Neighbor_v>> Reverse(const vector<priority_queue<Neighbor_v>>& B) {       //点vが近傍リストに入っている点のvectorを返す関数
    vector<vector<Neighbor_v>> R(B.size());
    for (int i = 0; i < B.size(); i++) {
        for (int j = 0; j < B.size(); j++) {
            priority_queue<Neighbor_v> B_copy = B[j];
            while (!B_copy.empty()) {
                Neighbor_v neighbor = B_copy.top();
                if (neighbor.num == i && i != j) {
                    neighbor.num = j;
                    R[i].push_back(neighbor);
                }
                B_copy.pop();
            }
        }
    }

    return R;
}

double hi_distance(const vector<double>& a, const vector<double>& b, int start, int end){ ////Histogram Intersectionでは値が大きいほうが似ているとなってしまうので1から引く
    double dist = 0.0;
    for (int i = start; i <= end; i++){
        dist += min(a[i],b[i]);
    }
    return 1-dist;
}

double l2_distance(const vector<double>& a, const vector<double>& b, int start, int end){       //L2距離
    double dist = 0.0;
    for (int i = start; i <= end; i++){
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(dist);
}

 

int isChanged(const priority_queue<Neighbor_v>& pq1, const priority_queue<Neighbor_v>& pq2) {

    if (pq1.size() != pq2.size()) {
        return 1;  // 要素の個数が異なる場合、変化したとみなす
    }

    priority_queue<Neighbor_v> pq1_copy = pq1;
    priority_queue<Neighbor_v> pq2_copy = pq2;
    int judge = 0;

    while (!pq1_copy.empty()) {
        Neighbor_v ele1 = pq1_copy.top();
        Neighbor_v ele2 = pq2_copy.top();
        pq1_copy.pop();
        pq2_copy.pop();
        
        // 距離の差が閾値以下ならば一致とみなす
        if (ele1.num == ele2.num && abs(ele1.dist - ele2.dist) > 10e-9) {
            judge = 1;
            break;
        }
    }

    if (judge == 1) {
        return 1;
    } else {
        return 0;
    }
}


int updateNN(priority_queue<Neighbor_v>& B, int u2, double l, int k){   //更新したいpriority_queue、近傍の近傍、近傍の近傍との距離、近傍の数
    priority_queue<Neighbor_v> pre_neighbor = B;
    priority_queue<Neighbor_v> change_neighbor;
    bool found = false;
    while(!B.empty()){  //近傍の近傍が存在すれば距離を更新する
        Neighbor_v neighbor = B.top();
        B.pop();

        if (neighbor.num == u2){
            neighbor.dist = l;
            found = true;
        }

        change_neighbor.push(neighbor);
    }
    if (!found) { //近傍に入ってなければ追加
        Neighbor_v new_neighbor;
        new_neighbor.num = u2;
        new_neighbor.dist = l;
        change_neighbor.push(new_neighbor);
    }

    while(B.size()<k){ //最大近傍数を超えないように追加
        Neighbor_v hold = change_neighbor.top();
        change_neighbor.pop();
        B.push(hold);
    }

    return isChanged(pre_neighbor, B);

}

vector<vector<Neighbor_v>> initializeUpdatedB(const vector<vector<Neighbor_v>>& B) {
    vector<vector<Neighbor_v>> initialized(B.size());
    for (int i = 0; i < B.size(); i++) {
        initialized[i] = vector<Neighbor_v>();
    }

    return initialized;
}

vector<vector<int>> create_udgraph(const vector<priority_queue<Neighbor_v>>& B){
    vector<vector<int>> graph(B.size(), vector<int>(B.size(), 0));
    for(int i=0; i < B.size(); i++){
        priority_queue<Neighbor_v> copy_neighbor = B[i];
        while(!copy_neighbor.empty()){
            graph[i][copy_neighbor.top().num] = 1;
            copy_neighbor.pop();
        }
    }
    return graph;
}

double Select_kth_ele(const priority_queue<Neighbor_v>& B, int k){
    priority_queue<Neighbor_v> queue_copy = B;
    double k_ele = 0.0;
    for(int i=0; i < k; i++){
        k_ele = queue_copy.top().dist;
        queue_copy.pop();
    }
    return k_ele;
}

void keep_q(priority_queue<Neighbor_v>& B){
    priority_queue<Neighbor_v> queue_copy;
    while(B.size() > 1){
        queue_copy.push(B.top());
        B.pop();
    }
    B = move(queue_copy);
}



bool Duplicate(priority_queue<Neighbor_v>& B, const Neighbor_v& newNode) {
    // 重複チェック
    bool dupcheck = false;
    priority_queue<Neighbor_v> queue_copy = B;
    while (!queue_copy.empty()) {
        Neighbor_v node = queue_copy.top();
        if (node.num == newNode.num) {
            dupcheck = true;
            break;
        }
        queue_copy.pop();
    }

    // 重複がない場合、新しい要素を追加
    if(dupcheck==false){
        B.push(newNode);
        return true;
    }else{
        return false;
    }
}




vector<vector<int>> Read_graph(const string& filename) {
    vector<vector<int>> result;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        vector<int> row;
        istringstream iss(line);
        string token;

        while (getline(iss, token, ',')) {
            int num;
            num = stoi(token);

            // Neighbor_v構造体を作成し、ベクトルに追加
            row.push_back(num);
        }
        result.push_back(row);
    }

    file.close();
    return result;
}

vector<vector<int>> Read_graph2(double& normalized, const string& filename) {
    vector<vector<int>> result;
    ifstream file(filename);

    string firstLine;
    getline(file, firstLine);
    normalized = std::stof(firstLine);

    string line;
    while (getline(file, line)) {
        vector<int> row;
        istringstream iss(line);
        string token;

        while (getline(iss, token, ',')) {
            int num;
            num = stoi(token);

            // Neighbor_v構造体を作成し、ベクトルに追加
            row.push_back(num);
        }
        result.push_back(row);
    }

    file.close();
    return result;
}

vector<density> Read_dent(const string& filename) {
    vector<density> row;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        string token;
        density key;
        getline(iss, token, ',');
        key.num = stoi(token);
        getline(iss, token, ',');
        key.dent = stof(token);
        row.push_back(key);
    }

    file.close();
    return row;
}

bool nn_check(const vector<int>& graph, int num){
    bool flag = false;
    for(int i=0; i < graph.size(); i++){
        if(graph[i] == num){
            flag = true;
        }
    }
    return flag;
}

//vector<vector<Neighbor_v>> select_start_node(const vector<density>& ,)

double calculate_alldist(const vector<vector<double>>& data, const vector<double>& weights, vector<vector<int>>& lange, vector<double> query, int num, double max1, double max2){
    double dist1 = hi_distance(query,data[num],lange[0][0],lange[0][1]);
    double dist2 = hi_distance(query,data[num],lange[1][0],lange[1][1]);
    double dist3 = l2_distance(query,data[num],lange[2][0],lange[2][1]) / max1;
    double dist4 = l2_distance(query,data[num],lange[3][0],lange[3][1]) / max2;
    double alldist = dist1 * weights[0] + dist2 * weights[1] + dist3 * weights[2] + dist4 * weights[3];

    return alldist;
}


vector<int> select_start_node(const vector<vector<double>>& data, const  vector<vector<density>>& centers, const vector<double>& weights, const vector<vector<int>>& lange, const vector<double>& query, const vector<int>& metrics){
    vector<int> start_node(5); //ID、グラフ番号（探索順）
    priority_queue<info> result;
    double l = 0.0;
    double max_dent = 0.0;
    double inf = numeric_limits<double>::infinity();
    for(int i=0; i < metrics.size(); i++){
        info min_dist = {0, i, 0.0, inf};
        if(metrics[i]==0){
            for(int j=0; j < centers[0].size(); j++){
                l = l2_distance(query,data[centers[i][j].num], lange[i][0], lange[i][1]);
                if(l < min_dist.dist){
                    min_dist.num = centers[i][j].num;
                    min_dist.dent = centers[i][j].dent * weights[i];
                    min_dist.dist = l;
                }
            }
        }

        if(metrics[i]==1){
            for(int k=0; k < centers[0].size(); k++){
                l = hi_distance(query,data[centers[i][k].num], lange[i][0], lange[i][1]);
                if(l < min_dist.dist){
                    min_dist.num = centers[i][k].num;
                    min_dist.dent = centers[i][k].dent * weights[i];
                    min_dist.dist = l;
                }
            }
        }
        result.push(min_dist);
    }

    start_node[0] = result.top().num;
    start_node[1] = result.top().graph;
    result.pop();
    int i=2;
    while(!result.empty()){
        start_node[i] = result.top().graph;
        result.pop();
        i++;
    }
    return start_node; //初期探索点、グラフ番号（探索順）
}

void writeCSV(const string& filename, const vector<vector<int>>& data) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "ファイルを開けませんでした。" << endl;
        return;
    }

    // データをCSV形式でファイルに書き込む
    for (const vector<int>& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ","; // カンマでデータを区切る
            }
        }
        file << endl; // 改行して次の行に移動
    }

    file.close();
    cout << "書き込み完了１" << endl;
}

void writeTimeQueryToCSV(const vector<chrono::duration<double>>& data, const string& filename) {
    ofstream file(filename);
    
    if (!file.is_open()) {
        cerr << "Failed to open the file: " << filename << endl;
        return;
    }

    // ベクトルの内容をCSVファイルに書き込む
    for (const auto& duration : data) {
        file << duration.count() << endl;
    }

    file.close();
    cout << "書き込み完了２" << endl;
}




int main(){
    double inf = numeric_limits<double>::infinity();
    double graph3_max = 0.0;
    double graph4_max = 0.0;


    //数値設定
    vector<vector<double>> query = read_embeddings("query_corel.asc");
    vector<double> weights = {0.2,0.2,0.3,0.3};
    vector<vector<int>> lange = {{0, 31}, {32, 63}, {64, 72}, {73, 88}};
    vector<int> metrics = {1,1,0,0}; //0はL2距離,1はHI
    vector<double> normalized = {1.0, 1.0, graph3_max, graph4_max};


    //グラフ読み込み
    vector<vector<double>> embeddings = read_embeddings("dataset_corel.asc");
    vector<vector<int>> graph1 = Read_graph("graph1_ColorHistogram.csv");

    vector<vector<int>> graph2 = Read_graph("graph2_LayoutHistogram.csv");
    vector<vector<int>> graph3 = Read_graph2(graph3_max, "graph3_ColorMoments.csv");
    vector<vector<int>> graph4 = Read_graph2(graph4_max, "graph4_CoocTexture.csv");
    vector<vector<vector<int>>> graph = {graph1, graph2, graph3, graph4};

    //代表点読み込み
    vector<density> centers1 = Read_dent("dent1_ColorHistogram.csv");
    vector<density> centers2 = Read_dent("dent2_LayoutHistogram.csv");
    vector<density> centers3 = Read_dent("dent3_ColorMoments.csv");
    vector<density> centers4 = Read_dent("dent4_CoocTexture.csv");
    vector<vector<density>> centers = {centers1, centers2, centers3 ,centers4};

    //クエリ結果
    vector<vector<int>> query_result(query.size(), vector<int>());
    vector<chrono::duration<double>> time_query(query.size());

    //探索
    for(int query_num; query_num < query.size(); query_num++){ //クエリの個数分繰り返し
        auto start_time = chrono::high_resolution_clock::now();
        vector<int> start_node = select_start_node(embeddings, centers, weights, lange, query[query_num], metrics); //初期探索点、グラフ探索順
        vector<int> visit_check(embeddings.size(), 0); //探索済みならば1に
        visit_check[start_node[0]] = 1;
        priority_queue<Neighbor_v> Candidate;
        priority_queue<Neighbor_v> Candidate_copy;
        Neighbor_v start_struct;
        double tau = inf;
        

        for(int metric=1; metric < start_node.size(); metric++){ //探索をグラフの個数回繰り返す
            if(weights[start_node[metric]]!=0.0){
                if(Candidate.empty()){
                    start_struct.num = start_node[0];
                    start_struct.dist = calculate_alldist(embeddings, weights, lange, query[query_num], start_node[0] , graph3_max, graph4_max);
                    Candidate.push(start_struct);
                }else{
                    start_struct = Candidate.top();
                }
                Candidate_copy.push(start_struct);

                while(!Candidate_copy.empty()){
                    Neighbor_v q_node = Candidate_copy.top();
                    Candidate_copy.pop();
                    for (int i=0; i < graph1[0].size(); i++){
                        if(graph[start_node[metric]][q_node.num][i]==1 && visit_check[i]!=1){ //q_node.numとエッジがつながっているかつ探索されていない      nn_check(graph[start_node[metric]][q_node.num])
                            visit_check[i] = 1;
                            if(Candidate.size() >= parameter_k+1){
                                tau = epsilon * Select_kth_ele(Candidate, parameter_k+1);
                            }
                            double dist = calculate_alldist(embeddings, weights, lange, query[query_num], i, graph3_max, graph4_max);
                            if(dist < tau){
                                Neighbor_v addednode;
                                addednode.num = i;
                                addednode.dist = dist;
                                bool addcheck = Duplicate(Candidate, addednode);
                                if(addcheck==true && Candidate.size() > parameter_k+1){
                                    keep_q(Candidate);
                                }
                                Duplicate(Candidate_copy, addednode);
                            }
                        }
                    }
                }
            }
        }

        auto end_time = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end_time - start_time;
        
        int i=0;
        while(i < parameter_k){
            query_result[query_num].push_back(Candidate.top().num);
            Candidate.pop();
            i++;
        }
        time_query[query_num] = duration;
    }
    writeCSV("result_corel_graph.csv", query_result);
    writeTimeQueryToCSV(time_query, "time_graph.csv");
    return 0;
}
