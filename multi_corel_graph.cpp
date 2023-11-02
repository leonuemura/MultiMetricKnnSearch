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
#define hp 2
#define max_line 1000  //変更
#define kmeans_chooselimit 5
#define center_limit 10

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

int random_search(int high){       //0からhighまでで乱数を発生
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dist(0,high);
    int num_random = dist(gen);
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

    while (!pq1_copy.empty() && !pq2_copy.empty()) {
        Neighbor_v ele1 = pq1_copy.top();
        Neighbor_v ele2 = pq2_copy.top();
        pq1_copy.pop();
        pq2_copy.pop();
        
        // 距離の差が閾値以下ならば一致とみなす
        if (ele1.num == ele2.num && abs(ele1.dist - ele2.dist) > 1e-9) {
        //if (ele1.num == ele2.num) {
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
    vector<vector<int>> graph(B.size(), vector<int>());
    for(int i=0; i < B.size(); i++){
        priority_queue<Neighbor_v> copy_neighbor = B[i];
        while(!copy_neighbor.empty()){
            graph[i].push_back(copy_neighbor.top().num);
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

bool findelement(const vector<int>& a, int rp_point) {
    return (find(a.begin(), a.end(), rp_point) != a.end());
}

bool containsValue(const vector<int> random, int value) {
    for (int i = 0; i < random.size(); i++) {
        if (random[i] == value) {
            return true; // 配列内に値が見つかった
        }
    }
    return false; // 配列内に値が見つからなかった
}


vector<int> kmeans_center(const vector<vector<double>>& data, double (*distance_func)(const vector<double>&, const vector<double>&, int, int), int start, int end) {  //k個の代表点を選ぶ
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> distribution(0.0, 1.0);
    vector<vector<int>> centroids(kmeans_chooselimit, vector<int>());
    vector<double> select_dist(kmeans_chooselimit);
    double inf = numeric_limits<double>::infinity();
    double dist = inf;

    // 最初のセントロイドをランダムに選択
    for(int m=0; m < kmeans_chooselimit; m++){ //kmeans_chooselimit回繰り返す
        cout << m << "周目" << endl;
        centroids[m].push_back(static_cast<int>(distribution(gen) * data.size())); //データセットの中からランダムに一つ選ぶ
        while (centroids[m].size() < center_limit) { //k個選ばれるまで続ける
            vector<double> distances(data.size(), numeric_limits<double>::max()); //距離を最大値で初期化
            for (int i = 0; i < data.size(); i++) {
                for (int j =0; j < centroids[m].size(); j++) {
                    double dist = distance_func(data[i], data[centroids[m][j]], start, end); //代表点との距離計算
                    distances[i] = min(distances[i], dist); //代表点の中で最も近いものを距離とする
                }
            }
            discrete_distribution<int> discreteDistribution(distances.begin(), distances.end());
            int rp_point = discreteDistribution(gen);
            if(containsValue(centroids[m], rp_point)==false){
                centroids[m].push_back(rp_point);
                cout << rp_point << endl;
            } else {
            }
        }
    }


    for(int m=0; m < kmeans_chooselimit; m++){
        for(int i=0; i < centroids[m].size(); i++){
            dist = inf;
            for(int j=0; j < centroids[m].size(); j++){
                if(i!=j){
                    dist = min(dist, distance_func(data[centroids[m][i]], data[centroids[m][j]], start, end));
                }
            }
            select_dist[m] += dist;
        }
    }
    cout << "代表点候補の計算終了選べました" << endl;

    double max_value = 0.0;
    int max_index = 0;
    for (int i = 0; i < select_dist.size(); i++) {
        if (select_dist[i] > max_value) {
            max_value = select_dist[i];
            max_index = i;
        }
    }
    cout << "代表点選べました" << endl;
    return centroids[max_index];
}

vector<density> density_add(const vector<priority_queue<Neighbor_v>>& data, const vector<int>& centers){  //代表点のk個目の近傍を使用し密度を持たせる関数
    vector<density> centroids;
    density center_dent;
    
    for(int i=0; i < centers.size(); i++){
        priority_queue<Neighbor_v> queue_copy = data[centers[i]];
        while(queue_copy.size() > 1){
            queue_copy.pop();
        }
        center_dent.num = centers[i];
        center_dent.dent = 1 / queue_copy.top().dist;
        centroids.push_back(center_dent);
        queue_copy.pop();
    }
    
    return centroids;
}



double L2_normalize(const vector<vector<double>>& data, int start, int end) {
    vector<vector<double>> minmax(2,vector<double>(end-start+1, 0.0));
    double minimum = 0.0;
    double maximum = 0.0;
    for (int i = start; i <= end; i++) {
        minimum = data[0][i];
        maximum = data[0][i];
        for (int j = 0; j < data.size(); j++) {
            minimum = min(minimum, data[j][i]);
            maximum = max(maximum, data[j][i]);
        }
        minmax[0][i-start] = minimum;
        minmax[1][i-start] = maximum;
    }
    
    double dist = 0.0;
    for (int i = 0; i < minmax[0].size(); i++){
        dist += (minmax[1][i] - minmax[0][i]) * (minmax[1][i] - minmax[0][i]);
    }

    return sqrt(dist);
}

void write_result_graph(const vector<vector<int>>& graph, const double& maxvalue, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "ファイルを開けませんでした。" << endl;
        return;
    }

    if(maxvalue != 0.0){
        file << maxvalue << endl;
    }

    for (const vector<int>& row : graph) {
        for (int i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << endl;
    }

    file.close();
}


void write_result_dent(const vector<density>& data, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "ファイルを開けませんでした。" << endl;
        return;
    }

    // データをCSVファイルに書き込む
    for (const density& center : data) {
        file << center.num << "," << center.dent << endl;
    }

    file.close();
}




int main(){
    auto start_time = chrono::high_resolution_clock::now();
    double l=0.0; //距離格納変数
    int c=0; //アップデートカウンタ
    
    double inf = numeric_limits<double>::infinity();
    vector<vector<double>> embeddings = read_embeddings("dataset_corel.asc"); //データベース
    vector<priority_queue<Neighbor_v>> B_heap1(embeddings.size(),  priority_queue<Neighbor_v>()); //グラフ1の近傍
    vector<priority_queue<Neighbor_v>> B_heap2(embeddings.size(),  priority_queue<Neighbor_v>()); //グラフ2の近傍
    vector<priority_queue<Neighbor_v>> B_heap3(embeddings.size(),  priority_queue<Neighbor_v>()); //グラフ3の近傍
    vector<priority_queue<Neighbor_v>> B_heap4(embeddings.size(),  priority_queue<Neighbor_v>()); //グラフ4の近傍
    vector<vector<Neighbor_v>> updated_B(B_heap1.size());  //各点の近傍を更新するための配列
    vector<vector<Neighbor_v>> R_vec;
    double graph3_max = L2_normalize(embeddings,64,72);
    double graph4_max = L2_normalize(embeddings,73,88);

    cout << "読み込み完了" << endl;

    


    for(int i=0; i<embeddings.size(); i++){
        vector<Neighbor_v> sample_result = sample(embeddings.size(), hp*parameter_k, i);  //各点の初期近傍をランダムにhp*parameter_k個選択
        for(int j=0; j < hp*parameter_k; j++){
            B_heap1[i].push(sample_result[j]);
            B_heap2[i].push(sample_result[j]);
            B_heap3[i].push(sample_result[j]);
            B_heap4[i].push(sample_result[j]);
        }
    }


    cout << "初期近傍選択完了" << endl;


    //グラフ1の作成(HI)
    while(1){
        R_vec = Reverse(B_heap1); //逆エッジ格納配列
        updated_B = initializeUpdatedB(updated_B);//更新近傍を初期化 
        
        for(int i = 0; i < B_heap1.size(); i++){
            priority_queue<Neighbor_v> B_heap_copy = B_heap1[i]; //2
            while(!B_heap_copy.empty()){
                Neighbor_v neighbor = B_heap_copy.top();
                updated_B[i].push_back(neighbor);    //近傍を更新近傍に追加
                B_heap_copy.pop();
            }

            for(const Neighbor_v& vec : R_vec[i]){  //更新近傍に逆近傍を追加
                bool duplicate = false;
                for(const Neighbor_v& existing : updated_B[i]){  //updated_Bの要素をexistingに追加
                    if(existing.num == vec.num){ //既に入っていれば追加しない
                        duplicate = true;
                        break;
                    }
                }
                if(!duplicate){
                    updated_B[i].push_back(vec);
                }
            }
        }
        

        c=0; //updateカウンター
        
        for(int i=0; i<updated_B.size(); i++){
            for(int j=0; j<updated_B[i].size(); j++){
                int neighbor_num = updated_B[i][j].num;  // 近傍の要素番号
                for(int k=0; k < updated_B[neighbor_num].size(); k++){
                    int neighbor_neighbor_num = updated_B[neighbor_num][k].num;  // 近傍の近傍の要素番号
                    if (i != neighbor_neighbor_num) {
                        l = hi_distance(embeddings[i], embeddings[neighbor_neighbor_num], 0, 31); //距離計算 //3
                        c += updateNN(B_heap1[i], neighbor_neighbor_num, l, hp*parameter_k); //近傍更新 //4
                    }
                }
            }
        }
        if(c==0) break; //更新されていなければ終了
        cout << "更新c:" << c << endl; 
    }

    vector<vector<int>> graph1 = create_udgraph(B_heap1);
    cout << "グラフ１作成完了" << endl;
    write_result_graph(graph1, 0.0, "graph1_ColorHistogram.csv");
    vector<int> center1 = kmeans_center(embeddings, hi_distance, 0, 31);
    vector<density> center_density1 = density_add(B_heap1, center1);
    cout << "グラフ１代表点選定完了" << endl;
    write_result_dent(center_density1, "dent1_ColorHistogram.csv");
    cout << "グラフ１出力完了" << endl;
    


    //グラフ2の作成(HI)
    while(1){
        R_vec = Reverse(B_heap2); //逆近傍
        updated_B = initializeUpdatedB(updated_B);//更新近傍を初期化

        for(int i = 0; i < B_heap2.size(); i++){ 
            priority_queue<Neighbor_v> B_heap_copy = B_heap2[i]; 
            while(!B_heap_copy.empty()){
                Neighbor_v neighbor = B_heap_copy.top();
                updated_B[i].push_back(neighbor);    //近傍を更新近傍に追加
                B_heap_copy.pop();
            }

            for(const Neighbor_v& vec : R_vec[i]){  //更新近傍に逆近傍を追加
                bool duplicate = false;
                for(const Neighbor_v& existing : updated_B[i]){  //updated_Bの要素をexistingに追加
                    if(existing.num == vec.num){ //既に入っていれば追加しない
                        duplicate = true;
                        break;
                    }
                }
                if(!duplicate){
                    updated_B[i].push_back(vec);
                }
            }
        }

        c=0;
        
        for(int i=0; i<updated_B.size(); i++){
            for(int j=0; j<updated_B[i].size(); j++){
                int neighbor_num = updated_B[i][j].num;  // 近傍の要素番号
                for(int k=0; k < updated_B[neighbor_num].size(); k++){
                    int neighbor_neighbor_num = updated_B[neighbor_num][k].num;  // 近傍の近傍の要素番号
                    if (i != neighbor_neighbor_num) {
                        l = hi_distance(embeddings[i], embeddings[neighbor_neighbor_num], 32, 63); //距離計算
                        c += updateNN(B_heap2[i], neighbor_neighbor_num, l, hp*parameter_k); //近傍更新
                    }
                }
            }
        }
        if(c==0) break; //更新されていなければ終了
        cout << "更新c:" << c << endl; 
    }

    vector<vector<int>> graph2 = create_udgraph(B_heap2);
    cout << "グラフ２作成完了" << endl;
    write_result_graph(graph2, 0.0, "graph2_LayoutHistogram.csv");
    vector<int> center2 = kmeans_center(embeddings, hi_distance, 32, 63);
    vector<density> center_density2 = density_add(B_heap2, center2); 
    cout << "グラフ２代表点選定完了" << endl;
    write_result_dent(center_density2, "dent2_LayoutHistogram.csv");
    cout << "グラフ２出力完了" << endl;
    




    //グラフ3の作成(L2)
    while(1){
        R_vec = Reverse(B_heap3); //逆近傍  
        updated_B = initializeUpdatedB(updated_B);//更新近傍を初期化

        for(int i = 0; i < B_heap3.size(); i++){
            priority_queue<Neighbor_v> B_heap_copy = B_heap3[i];
            while(!B_heap_copy.empty()){
                Neighbor_v neighbor = B_heap_copy.top();
                updated_B[i].push_back(neighbor);    //近傍を更新近傍に追加
                B_heap_copy.pop();
            }

            for(const Neighbor_v& vec : R_vec[i]){  //更新近傍に逆近傍を追加
                bool duplicate = false;
                for(const Neighbor_v& existing : updated_B[i]){  //updated_Bの要素をexistingに追加
                    if(existing.num == vec.num){ //既に入っていれば追加しない
                        duplicate = true;
                        break;
                    }
                }
                if(!duplicate){
                    updated_B[i].push_back(vec);
                }
            }
        }

        c=0; //updateカウンター
        
        for(int i=0; i<updated_B.size(); i++){
            for(int j=0; j<updated_B[i].size(); j++){
                int neighbor_num = updated_B[i][j].num;  // 近傍の要素番号
                for(int k=0; k < updated_B[neighbor_num].size(); k++){
                    int neighbor_neighbor_num = updated_B[neighbor_num][k].num;  // 近傍の近傍の要素番号
                    if (i != neighbor_neighbor_num) {
                        l = l2_distance(embeddings[i], embeddings[neighbor_neighbor_num], 64, 72) / graph3_max; //距離計算
                        c += updateNN(B_heap3[i], neighbor_neighbor_num, l, hp*parameter_k); //近傍更新
                    }
                }
            }
        }
        if(c==0) break; //更新されていなければ終了
        cout << "更新c:" << c << endl; 
    }

    vector<vector<int>> graph3 = create_udgraph(B_heap3);
    cout << "グラフ３作成完了" << endl;
    write_result_graph(graph3, graph3_max, "graph3_ColorMoments.csv");
    vector<int> center3 = kmeans_center(embeddings, l2_distance, 64, 72);
    vector<density> center_density3 = density_add(B_heap3, center3);
    cout << "グラフ３代表点選定完了" << endl;
    write_result_dent(center_density3, "dent3_ColorMoments.csv");
    cout << "グラフ３出力完了" << endl;




    //グラフ4の作成(L2)
    while(1){
        R_vec = Reverse(B_heap4); //逆近傍
        updated_B = initializeUpdatedB(updated_B);//更新近傍を初期化

        for(int i = 0; i < B_heap4.size(); i++){
            priority_queue<Neighbor_v> B_heap_copy = B_heap4[i]; 
            while(!B_heap_copy.empty()){
                Neighbor_v neighbor = B_heap_copy.top();
                updated_B[i].push_back(neighbor);    //近傍を更新近傍に追加
                B_heap_copy.pop();
            }

            for(const Neighbor_v& vec : R_vec[i]){  //更新近傍に逆近傍を追加
                bool duplicate = false;
                for(const Neighbor_v& existing : updated_B[i]){  //updated_Bの要素をexistingに追加
                    if(existing.num == vec.num){ //既に入っていれば追加しない
                        duplicate = true;
                        break;
                    }
                }
                if(!duplicate){
                    updated_B[i].push_back(vec);
                }
            }
        }

        c=0; //updateカウンター
        
        for(int i=0; i<updated_B.size(); i++){
            for(int j=0; j<updated_B[i].size(); j++){
                int neighbor_num = updated_B[i][j].num;  // 近傍の要素番号
                for(int k=0; k < updated_B[neighbor_num].size(); k++){
                    int neighbor_neighbor_num = updated_B[neighbor_num][k].num;  // 近傍の近傍の要素番号
                    if (i != neighbor_neighbor_num) {
                        l = l2_distance(embeddings[i], embeddings[neighbor_neighbor_num], 73, 88) / graph4_max; //距離計算
                        c += updateNN(B_heap4[i], neighbor_neighbor_num, l, hp*parameter_k); //近傍更新
                    }
                }
            }
        }
        if(c==0) break; //更新されていなければ終了
        cout << "更新c:" << c << endl; 
    }

    vector<vector<int>> graph4 = create_udgraph(B_heap4);
    cout << "グラフ４作成完了" << endl;
    write_result_graph(graph4, graph4_max, "graph4_CoocTexture.csv");
    vector<int> center4 = kmeans_center(embeddings, l2_distance, 73, 88);
    vector<density> center_density4 = density_add(B_heap4, center4);
    cout << "グラフ４代表点選定完了" << endl;
    write_result_dent(center_density4, "dent4_CoocTexture.csv");
    cout << "グラフ４出力完了" << endl;


    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end_time - start_time;

    cout << "実行時間: " << duration.count() << "秒" << endl;



    return 0;
}
